"""Render Ctrl-World's prediction against the camera at chosen episode steps.

Clean history throughout: the history handed to the model is the TRUE latent at
t-1, t-1-m, t-1-2m, ... so nothing here is contaminated by closed-loop drift.
Chunked exactly as deployment does -- one call emits K frames -- so frame t is
predicted ((t - SEED) %% K) + 1 steps past the newest history frame, and that
horizon is printed under each panel.

Frames before SEED = (num_history-1)*stride + 1 cannot be predicted at all: there
is no full history yet.  At the default 6 x stride 8 that is frame 41, so an early
step like 5 has no prediction to show.

    python experiments/stage2_twophase/predict_frames.py \
        --task pick_orange_cube_to_tray --episode 0 --frames 50 100 150 200 250
"""
import os, sys, pathlib, argparse, numpy as np, torch, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                       # noqa: E402
from vae_wrapper import VAEWrapper              # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="pick_orange_cube_to_tray")
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--demos_file", default="demos_disturbed.hdf5")
ap.add_argument("--frames", nargs="+", type=int, default=[5, 50, 100, 150, 200, 250])
ap.add_argument("--ckpt", default="outputs/stage1_stride_k8/stage1_rollout_ep1.pt")
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--out", default="outputs/diagnostics/clean_predict_steps.png")
ap.add_argument("--device", default="cuda:3")
a = ap.parse_args()

DEV = a.device
T_H = CONFIG["num_history"]; M = a.stride; K = a.K
SEED = (T_H - 1) * M + 1

p = os.path.join(REPO, "data", a.task, a.demos_file)
with h5py.File(p, "r") as f:
    e = sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit())))[a.episode]
    g = f[e]
    n_ep = len(g["observations_cam1"])
    want = sorted(set(a.frames))
    usable = [t for t in want if SEED <= t < n_ep]
    skipped = [t for t in want if t not in usable]
    n = min(n_ep, max(usable) + 1)
    imgs = np.array(g["observations_cam1"][:n])
    act = torch.from_numpy(np.array(g["actions"][:n + K], dtype=np.float32))
print(f"[ep] {a.task}/{e}  {n_ep} frames;  SEED={SEED}  K={K}  stride={M}")
if skipped:
    print(f"[ep] no prediction for frames {skipped}: "
          f"{'before SEED (no full history)' if min(skipped) < SEED else 'past end of episode'}")
if not usable:
    sys.exit("nothing to render")

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck
cw.eval()

x = torch.from_numpy(imgs).float().div_(255.).permute(0, 3, 1, 2)
with torch.no_grad():
    Z = torch.cat([vae.encode(x[j:j + 4].to(DEV)) for j in range(0, len(x), 4)])

# Chunked loop on CLEAN history, identical structure to rollout_collect.py.
zhat = {}
with torch.no_grad():
    for t in range(SEED, n, K):
        hidx = [t - 1 - i * M for i in range(T_H)][::-1]
        H = torch.stack([Z[j] for j in hidx], 0).unsqueeze(0)
        a_win = torch.cat([act[torch.tensor(hidx)], act[t:t + K]], 0).unsqueeze(0).to(DEV)
        zp = cw.predict_next_latent(H, a_win, n_steps=a.ddim).float()
        for i in range(min(K, n - t)):
            if t + i in usable:
                zhat[t + i] = zp[:, i]

dec = lambda z: vae.decode(z.to(DEV).float()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
cols = [t for t in usable if t in zhat]
fig, axes = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 6.4))
if len(cols) == 1:
    axes = axes.reshape(2, 1)
for c, t in enumerate(cols):
    h = (t - SEED) % K + 1
    axes[0, c].imshow(imgs[t]); axes[0, c].set_title(f"frame {t}\n(h={h} steps ahead)", fontsize=10)
    axes[1, c].imshow(np.clip(dec(zhat[t]), 0, 1))
    for r in (0, 1):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
axes[0, 0].set_ylabel("camera  $o_t$", fontsize=11)
axes[1, 0].set_ylabel("prediction  $\\hat{z}_t$", fontsize=11)
sub = f"{a.task}/{e} — clean history, K={K}, stride={M}, {a.ddim} DDIM steps, no Stage 2"
if skipped:
    sub += f"   (frames {skipped} have no full history: SEED={SEED})"
fig.suptitle(sub, fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.join(REPO, a.out)), exist_ok=True)
fig.savefig(os.path.join(REPO, a.out), dpi=130, bbox_inches="tight")
print(f"[fig] wrote {a.out}  columns={cols}")
