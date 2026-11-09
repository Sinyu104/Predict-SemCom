"""Closed-loop action check from WZ-SC closed-loop dumps (outputs/online_ft/old_nogate_*): per request, obs1/obs2 = TRUE camera images,
out1/out2 = received images the policy acted on, action_chunk = executed chunk.  Writes a score_arms_pi0fast.py npz:
real/wrist = true images (-> a*), arm 'vae' = RECEIVED images (the scorer requires an arm named vae), demo_a = executed action_chunk.

    python experiments/closed_loop_dump_arms.py --dump outputs/online_ft/old_nogate_snr_0dB_k4_30ep_11 --out <npz>
"""
import argparse, glob, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--task", default="pick up the blue cube and place it on the tray"); a = ap.parse_args()
fs = sorted(glob.glob(f"{a.dump}/*.npz"), key=lambda f: tuple(int(x) for x in re.findall(r"(\d+)_(\d+)\.npz", f)[0]))
out = {k: [] for k in ("real", "wrist", "img_vae", "wimg_vae", "demo_a", "task", "ep", "t", "n_obj", "n_mask")}
for f in fs:
    x = np.load(f); e, s = re.findall(r"(\d+)_(\d+)\.npz", f)[0]
    out["real"].append(x["obs1"]); out["wrist"].append(x["obs2"]); out["img_vae"].append(x["out1"]); out["wimg_vae"].append(x["out2"])
    out["demo_a"].append(x["action_chunk"]); out["task"].append(a.task); out["ep"].append(int(e)); out["t"].append(int(s))
    out["n_obj"].append(0); out["n_mask"].append(0)
np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, ckpt=a.dump, stage1="closed-loop dump")
print(a.out, len(fs))
