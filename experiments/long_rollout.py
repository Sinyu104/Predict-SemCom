"""Run the closed loop for a FULL episode and report the error curve.

Everything measured so far used 24-step rollouts, which is also all the Stage-1
fine-tune ever saw.  Episodes are ~240 steps, so whether the loop stays bounded
over a real episode is an extrapolation, not something the training data can
answer.  This runs the loop to the end of the episode and prints the error at
intervals, for one (stage1, stage2) pair.

    python experiments/long_rollout.py --ep 0 --tag new \
        --stage1 outputs/stage1_rollout_ft/stage1_rollout_ep3.pt \
        --stage2 outputs/stage2_warmstart/phase2.pt
"""
import argparse, pathlib, sys

import h5py
import numpy as np
import torch

REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from config import CONFIG                        # noqa: E402
from models import SemComSystem                  # noqa: E402
from vae_wrapper import VAEWrapper                # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--task", type=str, default="pick_red_cube_to_tray")
ap.add_argument("--demos_file", type=str, default="demos_disturbed.hdf5")
ap.add_argument("--tag", type=str, default="run")
ap.add_argument("--stage1", type=str, required=True)
ap.add_argument("--stage2", type=str, required=True)
ap.add_argument("--ddim_steps", type=int, default=10)
ap.add_argument("--channel", type=str, default="awgn")
ap.add_argument("--snr_db", type=float, default=20.0)
ap.add_argument("--max_steps", type=int, default=300)
a = ap.parse_args()

DEV = "cuda"
T_H = CONFIG["num_history"]
CONFIG["num_pred"] = 1
CONFIG["clip_length"] = T_H + 1
CONFIG["channel_type"] = a.channel
CONFIG["snr_db"] = a.snr_db

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV)
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=1,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(a.stage1, map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
cw.eval()

sysm = SemComSystem(CONFIG).to(DEV)
sk = torch.load(a.stage2, map_location="cpu")
sysm.load_state_dict(sk.get("system_state", sk), strict=False)
sysm.eval()

path = f"{REPO}/data/{a.task}/{a.demos_file}"
with h5py.File(path, "r") as f:
    e = sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit())))[a.ep]
    n = min(a.max_steps, len(f[e]["observations_cam1"]))
    x = torch.from_numpy(np.array(f[e]["observations_cam1"][:n])).float().div_(255.)
    x = x.permute(0, 3, 1, 2)
    act = torch.from_numpy(np.array(f[e]["actions"][:n], dtype=np.float32))

mse = lambda p, q: torch.nn.functional.mse_loss(p, q).item()
print(f"[{a.tag}] ep={a.ep} task={a.task} steps={n} "
      f"stage1={a.stage1.split('/')[-1]} stage2={a.stage2.split('/')[-1]}", flush=True)

with torch.no_grad():
    Z = torch.cat([vae.encode(x[j:j + 4].to(DEV)) for j in range(0, len(x), 4)])
    hist = [Z[k] for k in range(T_H)]
    torch.manual_seed(0)
    marks = [t for t in (6, 12, 24, 40, 60, 90, 120, 160, 200, 240, 280) if t < n]
    for t in range(T_H, n):
        H = torch.stack(hist[-T_H:], 0).unsqueeze(0)
        aw = act[t - T_H:t + 1].unsqueeze(0).to(DEV)
        zh = cw.predict_next_latent(H, aw, n_steps=a.ddim_steps)[:, 0].float()
        _, _, s = sysm.jscc_encoder(Z[t].unsqueeze(0), sample=False)
        st = sysm.channel(s)
        zt = sysm.refinement_diffusion.sdedit_refine(
            zh, st, noise_level=CONFIG.get("noise_level_second", 0), n_steps=a.ddim_steps)
        hist.append(zt[0])
        if t in marks:
            print(f"[{a.tag}] t={t:4d}  err(z~)={mse(zt, Z[t].unsqueeze(0)):.4f}  "
                  f"err(zhat)={mse(zh, Z[t].unsqueeze(0)):.4f}", flush=True)
print(f"[{a.tag}] DONE", flush=True)
