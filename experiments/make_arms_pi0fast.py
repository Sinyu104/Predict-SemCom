"""Stage A for the Pi0Fast sensitivity test (predictor env): build image arms at +30.

Arms (base camera only; the wrist camera is always the true frame):
  real = true frame, vae = VAE round trip of the true frame, pred = decoded +30 prediction
  (unit-noise start: blurry mean), predS = sigma-scaled noise start (a sharp sample),
  copy = decoded last history frame (stale).  Stage B: experiments/score_arms_pi0fast.py

For a few demo episodes, take the strided history ending at frame t-1 and the next
30 actions, predict the K=6 future latents (frames t-1+5j, j=1..6), VAE-decode them
and put them under the ORIGINAL camera frames at the same steps.

Column 0 is the last history frame (= what copy-last would output); columns 1..6 are
steps +5 ... +30.  Row A = original image, row B = prediction.  The last column is the
target the Wyner-Ziv receiver would actually use.

NOTE: every demo episode was seen in training (the train/val split was over samples,
not episodes), so this shows fit, not held-out generalisation.

    python experiments/viz_chunk30.py --ckpt outputs/stage1_chunk30/stage1_rollout_ep2.pt
"""
import argparse, glob, os, pathlib, sys

import h5py
import numpy as np
import torch
import torch.nn.functional as Fn

REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from config import CONFIG                        # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/stage1_chunk30/stage1_rollout_ep2.pt")
ap.add_argument("--cache", default="outputs/stage1_chunk30/latents.pt")
ap.add_argument("--demos", default="data/pick_*_cube_to_tray/demos.hdf5")
ap.add_argument("--out", default="outputs/stage1_chunk30/viz_ep2.png")
ap.add_argument("--episodes", type=int, nargs="+", default=[10, 150, 260, 380])
ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.7, 0.4],
                help="Position of t inside each episode's usable range.")
ap.add_argument("--ddim_steps", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--history_stride", type=int, default=10)
ap.add_argument("--views", type=int, default=1, help="2 = latent is base|wrist concatenated along width")
ap.add_argument("--out_npz", default="outputs/stage1_chunk30/pi0fast_arms.npz")
a = ap.parse_args()

DEV = "cuda"
T_H, M, PS, CHUNK = 6, a.history_stride, 5, 30
K = CHUNK // PS
SPAN = (T_H - 1) * M + 1
CONFIG["num_pred"] = K
torch.manual_seed(a.seed)

d = torch.load(os.path.join(REPO, a.cache), map_location="cpu")
Z = [z.float() for z in d["Z"]]; A = [x.float() for x in d["A"]]

# episode index -> (hdf5 file, key), in the same order the cache was built
index = []
for p in sorted(glob.glob(os.path.join(REPO, a.demos))):
    with h5py.File(p, "r") as f:
        index += [(p, k) for k in list(f) if "observations_cam1" in f[k]]
assert len(index) == len(Z), (len(index), len(Z))

cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
print("meta:", ck.get("meta"), "epoch:", ck.get("epoch"))
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV)

def hidx(t):
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]


def hidx(t):
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]
def u8(x):   # (3,H,W) float [0,1] -> (H,W,3) uint8
    return (x.permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)

rng = np.random.default_rng(a.seed)
out = {k: [] for k in ["real", "vae", "pred", "predS", "copy", "wrist", "demo_a", "ep", "t", "task",
                       "vae_w", "pred_w", "predS_w", "copy_w"]}
instr = {}
for i in range(a.n):
    e = int(rng.integers(len(Z))); T = len(Z[e])
    t = int(rng.integers(SPAN, T - CHUNK - CHUNK + 1))        # need CHUNK demo actions after frame t+29
    Ze, Ae = Z[e], A[e]
    f_tgt = t - 1 + CHUNK                                      # target frame index
    H = Ze[hidx(t)].unsqueeze(0).to(DEV)
    fut_a = torch.stack([Ae[t + PS * (j - 1): t + PS * j].mean(0) for j in range(1, K + 1)])
    ac = torch.cat([Ae[hidx(t)], fut_a], 0).unsqueeze(0).to(DEV)
    with torch.no_grad():
        pred = cw.predict_next_latent(H, ac, n_steps=a.ddim_steps).float()[0, -1]        # unit-noise start (mean-like, blurry)
        torch.manual_seed(a.seed + i)
        predS = cw.predict_next_latent(H, ac, n_steps=a.ddim_steps, init_scale=True).float()[0, -1]   # sigma-scaled start (sampled, sharp)
        dec = vae.decode(torch.stack([pred, Ze[f_tgt].to(DEV), H[0, -1], predS])).cpu()
    p, key = index[e]
    with h5py.File(p, "r") as f:
        real = np.array(f[key]["observations_cam1"][f_tgt])
        wrist = np.array(f[key]["observations_cam2"][f_tgt])
    tdir = os.path.dirname(p)
    if tdir not in instr:
        instr[tdir] = __import__("json").load(open(os.path.join(tdir, "task.json")))["instruction"]
    out["real"].append(real); out["wrist"].append(wrist)
    W1 = dec.shape[3] // a.views
    for j, nm in enumerate(["pred", "vae", "copy", "predS"]):
        out[nm].append(u8(dec[j][:, :, :W1]))
        if a.views == 2:
            out[nm + "_w"].append(u8(dec[j][:, :, W1:]))
    out["demo_a"].append(Ae[f_tgt:f_tgt + CHUNK].numpy()); out["ep"].append(e); out["t"].append(t)
    out["task"].append(instr[tdir])
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{a.n}", flush=True)
np.savez_compressed(os.path.join(REPO, a.out_npz),
                    **{k: np.array(v) for k, v in out.items() if len(v)}, ckpt=a.ckpt, views=a.views)
print("saved", a.out_npz, "DONE")
