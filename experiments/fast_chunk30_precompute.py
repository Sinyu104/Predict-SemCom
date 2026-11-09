"""Batched chunk-30 clean-history z_t / z_hat / z_prev precompute for Stage-2 (conv WZ codec).

Chunk-30 analogue of fast_clean_precompute.py.  The receiver gets one frame per 30-step
action chunk, so for each t (strided history ending at frame t-1, actions a_t..a_{t+29}):

    ZT    z[t-1+CHUNK]            true latent of the target frame
    ZH    Ctrl-World prediction of that frame (last of the K=CHUNK/PRED_STRIDE future slots)
    ZPREV z[t-1]                  transmitter's previous latent (WZ-clean mask reference)
    DIST  1 if any disturbed flag in [t-1, t+CHUNK-2]  (teleport applied after the last
          history frame and early enough to show in the target; flag is one frame early,
          so flag index f shows in obs f+1 -- see project_disturbance_label_offby1)

Output matches train_masked_alloc_wz.py (shard*.pt with ZT, ZH, ZPREV, DIST).

    # calm data (undisturbed demos, DIST=0; the predictor trained on these episodes):
    #   add  --demos_file demos.hdf5 --t_stride 8  and a different --out directory
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python experiments/fast_chunk30_precompute.py \
        --shard $i --nshards 4 --stage1_ckpt outputs/stage1_chunk30_h5/<ckpt>.pt \
        --out outputs/rollout_disturbed_chunk30/shard$i.pt &
    done; wait
"""
import argparse, os, pathlib, sys, time
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from config import CONFIG                       # noqa: E402
from vae_wrapper import VAEWrapper              # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--demos_file", default="disturbed.hdf5")
ap.add_argument("--stage1_ckpt", required=True)
ap.add_argument("--history_stride", type=int, default=5, help="M; must match the predictor's training")
ap.add_argument("--chunk", type=int, default=30)
ap.add_argument("--pred_stride", type=int, default=5)
ap.add_argument("--t_stride", type=int, default=1, help="keep every n-th t (neighbouring chunk windows overlap 29/30)")
ap.add_argument("--views", type=int, default=1, help="2 = base+wrist latents concatenated along width (4x28x56), as the 2-cam predictor")
ap.add_argument("--holdout_mod", type=int, default=0, help="skip episodes whose index %% holdout_mod == 0 (kept for scoring)")
ap.add_argument("--max_eps_per_task", type=int, default=0, help="keep only the first N episodes of each task (0 = all)")
ap.add_argument("--init_scale", action="store_true", help="sigma-scaled noise start (sharp sample)")
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--pred_bs", type=int, default=16)
ap.add_argument("--vae_bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = "cuda:0"
T_H = CONFIG["num_history"]; M = a.history_stride; PS = a.pred_stride; CH = a.chunk
K = CH // PS
SPAN = (T_H - 1) * M + 1                 # earliest t with a full strided history
CONFIG["num_pred"] = K
torch.manual_seed(a.seed + a.shard)

index = []
for task in a.tasks:
    p = os.path.join(REPO, "data", task, a.demos_file)
    if not os.path.isfile(p):
        print(f"[shard {a.shard}] skip {task}: no {a.demos_file}", flush=True); continue
    with h5py.File(p, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit())))
        if a.max_eps_per_task: keys = keys[:a.max_eps_per_task]
        for e in keys:
            if a.views == 2 and "observations_cam2" not in f[e]: continue
            if a.holdout_mod and int("".join(c for c in e if c.isdigit())) % a.holdout_mod == 0: continue
            n = len(f[e]["observations_cam1"])
            for t in range(SPAN, n - CH + 1, a.t_stride):          # needs actions a_t..a_{t+CH-1}
                index.append((p, e, t))
index = index[a.shard::a.nshards]
print(f"[shard {a.shard}] {len(index)} frames, T_H={T_H} M={M} K={K} chunk={CH} ddim={a.ddim}", flush=True)

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); print("ckpt meta:", ck.get("meta")); del ck; cw.eval()

enc_cache = {}
def episode(p, e):
    key = (p, e)
    if key not in enc_cache:
        enc_cache.clear()
        with h5py.File(p, "r") as f:
            g = f[e]
            x = torch.from_numpy(np.array(g["observations_cam1"])).float().div_(255.).permute(0, 3, 1, 2)
            act = torch.from_numpy(np.array(g["actions"], dtype=np.float32))
            dist = (np.array(g["disturbed"]).astype(np.float32) if "disturbed" in g
                    else np.zeros(len(x), dtype=np.float32))
        enc = lambda xx: torch.cat([vae.encode(xx[j:j + a.vae_bs].to(DEV)).cpu() for j in range(0, len(xx), a.vae_bs)])
        with torch.no_grad():
            Z = enc(x)
            if a.views == 2:
                with h5py.File(p, "r") as f2:
                    x2 = torch.from_numpy(np.array(f2[e]["observations_cam2"])).float().div_(255.).permute(0, 3, 1, 2)
                Z = torch.cat([Z, enc(x2)], dim=-1)                       # (T,4,28,56)
        enc_cache[key] = (Z, act, dist)
    return enc_cache[key]

def hidx(t):
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]     # oldest first

ZT, ZH, ZP, DIST = [], [], [], []
t0 = time.perf_counter(); i = 0; nb = 0
while i < len(index):
    p, e, _ = index[i]
    Z, act, dist = episode(p, e)
    batch = []
    while i < len(index) and index[i][0] == p and index[i][1] == e and len(batch) < a.pred_bs:
        batch.append(index[i][2]); i += 1
    H = torch.stack([Z[hidx(t)] for t in batch]).to(DEV)                          # (B,T_H,C,h,w)
    # history actions + K future slots, each the mean of PS actions (same as make_arms_pi0fast)
    aw = torch.stack([torch.cat([act[hidx(t)],
                                 torch.stack([act[t + PS * (j - 1): t + PS * j].mean(0) for j in range(1, K + 1)])], 0)
                      for t in batch]).to(DEV)
    with torch.no_grad():
        zp = cw.predict_next_latent(H, aw, n_steps=a.ddim, init_scale=a.init_scale).float()[:, -1]
    ZH.append(zp.half().cpu())
    ZT.append(torch.stack([Z[t - 1 + CH] for t in batch]).half())
    ZP.append(torch.stack([Z[t - 1] for t in batch]).half())
    DIST.append(torch.tensor([float(dist[t - 1: t - 1 + CH].max() > 0.5) for t in batch]))
    nb += 1
    if nb % 20 == 0:
        done = sum(len(z) for z in ZT); el = time.perf_counter() - t0
        print(f"[shard {a.shard}] {done}/{len(index)}  {el/done*1000:.0f} ms/frame  "
              f"eta {el/done*(len(index)-done)/60:.0f} min", flush=True)

ZT = torch.cat(ZT); ZH = torch.cat(ZH); ZP = torch.cat(ZP); DIST = torch.cat(DIST)
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
torch.save({"ZT": ZT, "ZH": ZH, "ZPREV": ZP, "DIST": DIST,
            "meta": {"chunk": CH, "history_stride": M, "pred_stride": PS, "ddim": a.ddim,
                     "init_scale": a.init_scale, "views": a.views, "holdout_mod": a.holdout_mod, "t_stride": a.t_stride, "max_eps_per_task": a.max_eps_per_task, "stage1": a.stage1_ckpt, "demos_file": a.demos_file}}, a.out)
f = lambda x, y: ((x.float() - y.float()) ** 2).mean().item()
m = (((ZT.float() - ZP.float()) ** 2).mean(1) > 0.05).float().mean().item()   # over the whole (possibly 2-view) grid
print(f"[shard {a.shard}] saved {a.out} ZT={tuple(ZT.shape)} disturbed={int(DIST.sum())}/{len(DIST)} "
      f"|zt-zhat|^2={f(ZT, ZH):.4f} |zt-zprev|^2={f(ZT, ZP):.4f} mask_density(thr .05)={m:.3f}", flush=True)
