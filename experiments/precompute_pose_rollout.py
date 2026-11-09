"""ZT / ZH / ZPREV cache for Stage-2 from the POSE-ONLY chunk-30 predictor (oracle pose of every history + future frame), built on the
policy-rollout latents (outputs/stage1_chunk30_2cam_ft/rollout_latents.pt, 500 episodes, no disturbance, DIST = 0).

Same outputs as fast_chunk30_precompute.py (ZT, ZH, ZPREV, DIST) plus HELD (1 = episode held out of the predictor's own training, so its
ZH is a genuine out-of-sample prediction) and EP.

    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$g python experiments/precompute_pose_rollout.py --shard $i --nshards 3 \
        --ckpt outputs/stage1_chunk30_pose_unfreeze/stage1_rollout_ep4.pt --t_stride 20 --out outputs/rollout_pose_unf4/shard$i.pt & done
"""
import argparse, glob, os, pathlib, sys, time
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache", default="outputs/stage1_chunk30_2cam_ft/rollout_latents.pt")
ap.add_argument("--demos", default="data/pick_*_cube_to_tray/policy_rollouts.hdf5")
ap.add_argument("--holdout_per_file", type=int, default=10)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--t_stride", type=int, default=20)
ap.add_argument("--t_offset", type=int, default=0, help="start offset of the window grid (use another value for an extra, interleaved set)")
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--pred_bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--held_only", action="store_true", help="only the predictor's held-out episodes")
ap.add_argument("--init_scale", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda"; T_H, M, PS, CH = 6, 5, 5, 30; K = CH // PS; SPAN = (T_H - 1) * M + 1
CONFIG["num_pred"] = K
torch.manual_seed(a.seed + a.shard)

d = torch.load(os.path.join(REPO, a.cache), map_location="cpu")
Z = d["Z"]; A = d["A"]; FILE = list(d["F"]); n_ep = len(Z)
rs = np.random.default_rng(123); held = set()                        # same split as the predictor's training (stage1_rollout_finetune)
for fi in sorted(set(FILE)):
    idx = [e for e in range(n_ep) if FILE[e] == fi]
    held |= set(int(x) for x in rs.choice(idx, min(a.holdout_per_file, len(idx)), replace=False))
P_RAW = []
for p_ in sorted(glob.glob(os.path.join(REPO, a.demos))):
    with h5py.File(p_, "r") as f_:
        for e_ in list(f_):
            if "observations_cam1" in f_[e_]:
                g_ = f_[e_]
                P_RAW.append(torch.from_numpy(np.concatenate([np.array(g_["robot_ee_pos"]), np.array(g_["robot_joint_pos"])], 1)).float())
assert len(P_RAW) == n_ep and all(len(P_RAW[e]) == len(Z[e]) for e in range(n_ep))

ck = torch.load(os.path.join(REPO, a.ckpt), map_location="cpu"); meta = ck["meta"]
POSE = bool(meta.get("pose_only"))        # else: action-conditioned predictor (7-dim)
cw = CtrlWorldWrapper(action_dim=meta["pose_dim"] if POSE else CONFIG["action_dim"], num_history=T_H, num_pred=K, svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=True, extra_unfreeze=tuple(meta.get("unfreeze", [])), dtype=torch.float16).to(DEV)
cw.action_encoder.load_state_dict(ck["action_encoder_state"]); cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
if "unet_extra_state" in ck: cw.load_unet_extra_state_dict(ck["unet_extra_state"])
cw.eval(); print("ckpt meta:", {k: v for k, v in meta.items() if k not in ("pose_mean", "pose_std")}, flush=True)
if POSE: mean = torch.tensor(meta["pose_mean"]); std = torch.tensor(meta["pose_std"]); sc = meta.get("pose_scale", 1.0)
hidx = lambda t: [max(0, t - 1 - i * M) for i in range(T_H)][::-1]
fidx = lambda t: [t - 1 + PS * j for j in range(1, K + 1)]

index = [(e, t) for e in range(n_ep) for t in range(SPAN + a.t_offset, len(Z[e]) - CH + 1, a.t_stride) if (e in held or not a.held_only)][a.shard::a.nshards]
print(f"[shard {a.shard}] {len(index)} windows ({sum(1 for e, _ in index if e in held)} from predictor-held-out episodes)", flush=True)

ZT, ZH, ZP, EP = [], [], [], []
t0 = time.perf_counter(); i = 0; nb = 0
while i < len(index):
    batch = index[i:i + a.pred_bs]; i += len(batch)
    H = torch.stack([Z[e][hidx(t)].float() for e, t in batch]).to(DEV)
    if POSE:
        pe = torch.stack([torch.cat([(P_RAW[e][hidx(t) + fidx(t)] - mean) / std * sc, torch.ones(T_H + K, 1)], 1) for e, t in batch]).to(DEV)
    else:   # history actions + K future slots, each the mean of PS actions
        pe = torch.stack([torch.cat([A[e][hidx(t)].float(), torch.stack([A[e][t + PS * (j - 1): t + PS * j].float().mean(0) for j in range(1, K + 1)])], 0) for e, t in batch]).to(DEV)
    with torch.no_grad():
        zp = cw.predict_next_latent(H, pe, n_steps=a.ddim, init_scale=bool(a.init_scale)).float()[:, -1]
    ZH.append(zp.half().cpu()); ZT.append(torch.stack([Z[e][t - 1 + CH] for e, t in batch])); ZP.append(torch.stack([Z[e][t - 1] for e, t in batch]))
    EP.append(torch.tensor([e for e, _ in batch])); nb += 1
    if nb % 20 == 0:
        el = time.perf_counter() - t0
        print(f"[shard {a.shard}] {i}/{len(index)}  {el/i*1000:.0f} ms/frame  eta {el/i*(len(index)-i)/60:.0f} min", flush=True)

ZT = torch.cat(ZT); ZH = torch.cat(ZH); ZP = torch.cat(ZP); EP = torch.cat(EP)
HELD = torch.tensor([float(int(e) in held) for e in EP])
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
torch.save({"ZT": ZT, "ZH": ZH, "ZPREV": ZP, "DIST": torch.zeros(len(ZT)), "HELD": HELD, "EP": EP,
            "meta": {"chunk": CH, "history_stride": M, "pred_stride": PS, "ddim": a.ddim, "init_scale": bool(a.init_scale), "views": 2, "pose_only": POSE,
                     "t_stride": a.t_stride, "t_offset": a.t_offset, "stage1": a.ckpt, "demos_file": "policy_rollouts.hdf5"}}, a.out)
f = lambda x, y: ((x.float() - y.float()) ** 2).mean().item()
m = (((ZT.float() - ZP.float()) ** 2).mean(1) > 0.05).float().mean().item()
h = HELD.bool()
print(f"[shard {a.shard}] saved {a.out} ZT={tuple(ZT.shape)} |zt-zhat|^2={f(ZT, ZH):.4f} (held-out eps {f(ZT[h], ZH[h]) if h.any() else float('nan'):.4f}) "
      f"|zt-zprev|^2={f(ZT, ZP):.4f} mask_density(thr .05)={m:.3f}", flush=True)
