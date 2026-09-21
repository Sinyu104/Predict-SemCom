"""Fast BATCHED clean-history z_t / z_hat precompute for Stage-2 training.

rollout_collect.py runs Ctrl-World sequentially, one frame at a time, because in the
closed loop each frame conditions on the PREVIOUS reconstruction.  For clean history
(reanchor=1) that dependency is gone: the prediction for frame t uses the TRUE latents
at t-1..t-num_history, so every frame is independent and they can be BATCHED.  The
sequential loop left the GPU ~95%% idle (measured ~1500-2600 s/episode); batching the
predictions is 10-50x faster.

Output matches rollout_to_stage2's s2 format directly (ZT, ZH, DIST), so twophase.py
consumes it as-is:
    ZT   (N, 4, 28, 28)  true latent z_t
    ZH   (N, 4, 28, 28)  clean-history prediction z_hat_t
    DIST (N,)            per-frame disturbance flag (from disturbed.hdf5)

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python experiments/fast_clean_precompute.py \
        --shard $i --nshards 4 --demos_file disturbed.hdf5 \
        --stage1_ckpt outputs/stage1_rollout_ft/stage1_rollout_ep1.pt \
        --out outputs/rollout_disturbed_clean_s2/shard$i.pt &
    done; wait
"""
import argparse, glob, os, pathlib, sys, time
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
ap.add_argument("--stage1_ckpt", default="outputs/stage1_rollout_ft/stage1_rollout_ep1.pt")
ap.add_argument("--history_stride", type=int, default=1, help="m; clean history spacing")
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--pred_bs", type=int, default=32, help="frames predicted per Ctrl-World call")
ap.add_argument("--vae_bs", type=int, default=16)
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = "cuda:0"          # one GPU per process via CUDA_VISIBLE_DEVICES
T_H = CONFIG["num_history"]; M = a.history_stride
SPAN = (T_H - 1) * M + 1                 # earliest frame with a full strided history

# ── enumerate every predictable frame (episode, t) across this shard ────────── #
index = []
for task in a.tasks:
    p = os.path.join(REPO, "data", task, a.demos_file)
    if not os.path.isfile(p):
        print(f"[shard {a.shard}] skip {task}: no {a.demos_file}", flush=True); continue
    with h5py.File(p, "r") as f:
        for e in sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit()))):
            n = len(f[e]["observations_cam1"])
            for t in range(SPAN, n):
                index.append((p, e, t))
index = index[a.shard::a.nshards]
print(f"[shard {a.shard}] {len(index)} frames, T_H={T_H} stride={M} ddim={a.ddim}", flush=True)

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=1,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()

# ── cache per-episode latents once (batched VAE), reused across that episode's frames #
enc_cache = {}     # (p,e) -> (Z tensor on CPU, actions tensor, disturbed array)
def episode(p, e):
    key = (p, e)
    if key not in enc_cache:
        enc_cache.clear()                        # only ever need the current episode
        with h5py.File(p, "r") as f:
            g = f[e]
            x = torch.from_numpy(np.array(g["observations_cam1"])).float().div_(255.).permute(0, 3, 1, 2)
            act = torch.from_numpy(np.array(g["actions"], dtype=np.float32))
            dist = (np.array(g["disturbed"]).astype(np.float32) if "disturbed" in g
                    else np.zeros(len(x), dtype=np.float32))
        with torch.no_grad():
            Z = torch.cat([vae.encode(x[j:j + a.vae_bs].to(DEV)).cpu()
                           for j in range(0, len(x), a.vae_bs)])
        enc_cache[key] = (Z, act, dist)
    return enc_cache[key]

def hidx(t):
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]     # oldest first

ZT, ZH, DIST = [], [], []
t0 = time.perf_counter()
# group consecutive frames of the SAME episode into prediction batches
i = 0
while i < len(index):
    p, e, _ = index[i]
    Z, act, dist = episode(p, e)
    # gather a batch of frames from this same episode
    batch = []
    while i < len(index) and index[i][0] == p and index[i][1] == e and len(batch) < a.pred_bs:
        batch.append(index[i][2]); i += 1
    H  = torch.stack([torch.stack([Z[j] for j in hidx(t)], 0) for t in batch]).to(DEV)   # (B,T_H,C,H,W)
    # action window per frame: strided history actions + the current action (num_pred=1)
    aw = torch.stack([torch.cat([act[torch.tensor(hidx(t))], act[t:t + 1]], 0) for t in batch]).to(DEV)
    with torch.no_grad():
        zp = cw.predict_next_latent(H, aw, n_steps=a.ddim).float()   # (B,1,C,H,W)
    ZH.append(zp[:, 0].half().cpu())
    ZT.append(torch.stack([Z[t] for t in batch]).half())
    DIST.append(torch.tensor([dist[t] for t in batch]))
    if len(ZT) % 20 == 0:
        done = sum(len(z) for z in ZT); el = time.perf_counter() - t0
        print(f"[shard {a.shard}] {done}/{len(index)}  {el/done*1000:.0f} ms/frame  "
              f"eta {el/done*(len(index)-done)/60:.0f} min", flush=True)

ZT = torch.cat(ZT); ZH = torch.cat(ZH); DIST = torch.cat(DIST)
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
torch.save({"ZT": ZT, "ZH": ZH, "DIST": DIST,
            "meta": {"clean_history": True, "history_stride": M, "ddim": a.ddim,
                     "stage1": a.stage1_ckpt, "demos_file": a.demos_file}}, a.out)
dev = ((ZT.float() - ZH.float()) ** 2).mean().item()
print(f"[shard {a.shard}] saved {a.out}  ZT={tuple(ZT.shape)}  "
      f"disturbed={int(DIST.sum())}/{len(DIST)}  mean||z-zhat||^2={dev:.4f}", flush=True)
