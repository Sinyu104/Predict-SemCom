"""SELF-HISTORY ZT / ZH / ZPREV cache for Stage-2: the predictor is chained over each episode exactly as in the closed loop
(server/semcom_server.py _step_wz2cam), instead of predicting every window from TRUE history (precompute_pose_rollout.py).

Per episode (policy-rollout latents, oracle pose rows), with base = 30(n-1) the last history frame of request n:
    request 0 (step 0) anchor, request 1 (step 30) warm start: history = TRUE frames at steps 5,10,..,30
    request n >= 2:  F = Ctrl-World(H, pose)            6 predicted frames base+5 .. base+30;  zhat = F[-1]
                     z~ = codec(z_t, zhat) at k channels over CDL-C at this batch's SNR (the current codec, as deployed)
                     next H = [F[0..4], z~]             in-between slots stay predicted, only the boundary slot is corrected
So ZH carries the closed-loop error (drift of the predicted in-between history).  Stored per sample: ZT = z_{base+30},
ZPREV = z_base, ZH, SNR (the loop SNR), N (request index), EP, HELD.  DIST = 0 (no disturbance in these rollouts).

    CUDA_VISIBLE_DEVICES=0 python experiments/precompute_pose_selfhist.py --shard 0 --nshards 2 \
        --ckpt outputs/stage1_chunk30_pose_unfreeze2/stage1_rollout_ep5.pt --codec outputs/wz2cam_retrain/K_nogate_snr/masked_alloc.pt \
        --out outputs/rollout_pose_unf2_selfhist/shard0.pt
"""
import argparse, glob, os, pathlib, sys, time
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/stage2_twophase"))
from config import CONFIG                        # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402
from models import FadingChannel                 # noqa: E402
from conv_codec import ConvWZCodec               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--codec", required=True, help="Stage-2 codec in the loop (makes the corrected boundary slot z~)")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--snr_min", type=float, default=0.0)
ap.add_argument("--snr_max", type=float, default=20.0)
ap.add_argument("--cache", default="outputs/stage1_chunk30_2cam_ft/rollout_latents.pt")
ap.add_argument("--demos", default="data/pick_*_cube_to_tray/policy_rollouts.hdf5")
ap.add_argument("--holdout_per_file", type=int, default=10)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--pred_bs", type=int, default=4, help="episodes chained together (one SNR per group)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda"; T_H, PS, CH = 6, 5, 30; K = CH // PS
CONFIG["num_pred"] = K
torch.manual_seed(a.seed + a.shard); rng = np.random.default_rng(a.seed + a.shard)

d = torch.load(os.path.join(REPO, a.cache), map_location="cpu")
Z = d["Z"]; FILE = list(d["F"]); n_ep = len(Z)
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
assert meta.get("pose_only"), "self-history cache is written for the pose-only predictor"
cw = CtrlWorldWrapper(action_dim=meta["pose_dim"], num_history=T_H, num_pred=K, svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=True, extra_unfreeze=tuple(meta.get("unfreeze", [])), dtype=torch.float16).to(DEV)
cw.action_encoder.load_state_dict(ck["action_encoder_state"]); cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
if "unet_extra_state" in ck: cw.load_unet_extra_state_dict(ck["unet_extra_state"])
cw.eval()
mean = torch.tensor(meta["pose_mean"]); std = torch.tensor(meta["pose_std"]); sc = meta.get("pose_scale", 1.0)

cc = torch.load(os.path.join(REPO, a.codec), map_location="cpu"); c = cc["cfg"]
assert c.get("mask_mode") == "none" and c.get("residual"), "loop codec must be a no-mask residual codec (z~ = rec)"
net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"], residual=True, snr_cond=bool(c.get("snr_cond", False))).to(DEV).eval()
net.load_state_dict(cc["net"])
ch = FadingChannel(20.0, channel_type="cdl", cdl_kwargs=dict(model="C", carrier_frequency=3.5e9, max_speed=3.0, delay_spread=50e-9))

eps = list(range(n_ep))[a.shard::a.nshards]
n_req = lambda e: (len(Z[e]) - 1) // CH                                # requests n = 0 .. n_req (frame 30n must exist)
total = sum(max(0, n_req(e) - 1) for e in eps)
print(f"[shard {a.shard}] {len(eps)} episodes, {total} chained predictions, loop codec {a.codec} k={a.k} SNR U[{a.snr_min},{a.snr_max}]", flush=True)

ZT, ZH, ZP, EP, SNR, NN = [], [], [], [], [], []
t0 = time.perf_counter(); done = 0
for g in range(0, len(eps), a.pred_bs):
    grp = eps[g:g + a.pred_bs]
    snr = float(rng.uniform(a.snr_min, a.snr_max)); ch.snr_db = snr
    H = {e: Z[e][[PS * i for i in range(1, T_H + 1)]].float().to(DEV) for e in grp}   # warm start: true frames 5..30
    n = 2
    while True:
        act = [e for e in grp if n <= n_req(e)]
        if not act: break
        base = CH * (n - 1)
        idx = [max(0, base - PS * i) for i in range(T_H)][::-1] + [base + PS * j for j in range(1, K + 1)]
        pe = torch.stack([torch.cat([(P_RAW[e][idx] - mean) / std * sc, torch.ones(len(idx), 1)], 1) for e in act]).to(DEV)
        with torch.no_grad():
            F = cw.predict_next_latent(torch.stack([H[e] for e in act]), pe, n_steps=a.ddim, init_scale=True).float()   # (B,K,4,28,56)
            zh = F[:, -1]; zt = torch.stack([Z[e][base + CH] for e in act]).float().to(DEV)
            mu_e, _ = net.encode(zt); mu_p, _ = net.prior(zh); B = len(act)
            s_t = ch(mu_e[:, :a.k].flatten(1)).view(B, a.k, *mu_e.shape[2:])
            zz = net.decode(torch.cat([s_t, mu_p[:, a.k:]], 1), zh, None, snr)
        for b, e in enumerate(act):
            H[e] = torch.cat([F[b, :-1], zz[b:b + 1]], 0)              # next history: predicted in-between slots + corrected boundary
        ZH.append(zh.half().cpu()); ZT.append(zt.half().cpu()); ZP.append(torch.stack([Z[e][base] for e in act]))
        EP.append(torch.tensor(act)); SNR.append(torch.full((B,), snr)); NN.append(torch.full((B,), n))
        done += B; n += 1
    el = time.perf_counter() - t0
    print(f"[shard {a.shard}] {done}/{total}  {el/max(done,1)*1000:.0f} ms/frame  eta {el/max(done,1)*(total-done)/60:.0f} min", flush=True)

ZT = torch.cat(ZT); ZH = torch.cat(ZH); ZP = torch.cat(ZP); EP = torch.cat(EP); SNR = torch.cat(SNR); NN = torch.cat(NN)
HELD = torch.tensor([float(int(e) in held) for e in EP])
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
torch.save({"ZT": ZT, "ZH": ZH, "ZPREV": ZP, "DIST": torch.zeros(len(ZT)), "HELD": HELD, "EP": EP, "SNR": SNR, "N": NN,
            "meta": {"chunk": CH, "history_stride": PS, "pred_stride": PS, "ddim": a.ddim, "init_scale": True, "views": 2, "pose_only": True,
                     "self_history": True, "loop_codec": a.codec, "loop_k": a.k, "snr_range": [a.snr_min, a.snr_max], "stage1": a.ckpt,
                     "demos_file": "policy_rollouts.hdf5"}}, a.out)
f = lambda x, y: ((x.float() - y.float()) ** 2).mean().item()
lo = SNR < 10
print(f"[shard {a.shard}] saved {a.out} ZT={tuple(ZT.shape)} |zt-zhat|^2={f(ZT, ZH):.4f} (SNR<10: {f(ZT[lo], ZH[lo]):.4f}, SNR>=10: {f(ZT[~lo], ZH[~lo]):.4f}) "
      f"by request n: " + " ".join(f"{int(v)}:{f(ZT[NN == v], ZH[NN == v]):.3f}" for v in NN.unique()), flush=True)
