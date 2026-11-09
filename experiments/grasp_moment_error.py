"""GRASP-split offline action error: windows whose 30-step chunk on the TRUE images (a*, PI0Fast) contains a gripper open<->close
transition (the grasp/release moment).  Same 200 held-out windows for every method; SSCC counts received windows only.
Appends rows (split GRASP) to results/wz_offline_pi0fast.csv:  a_real_xyz, a_real_xyz_se, grip_wrong (fraction of steps whose
open/close command differs from a*).

    python experiments/grasp_moment_error.py
"""
import csv, datetime, glob, os, subprocess
import numpy as np
D = "outputs/cmp_pose_unf2"; CSV = "results/wz_offline_pi0fast.csv"
SRC = [("nogate", None), ("nogate_k8dB", None), ("dvst", None), ("sscc", "sscc800"), ("sscc_match/k16", "sscc_k16")]

def load(sub):
    parts = sorted(glob.glob(f"{D}/{sub}/scores_part*_actions.npz")) or [f"{D}/{sub}/scores_actions.npz"]
    arms = sorted(glob.glob(f"{D}/{sub}/arms_part*.npz")) or [f"{D}/{sub}/arms.npz"]
    act, keys, ok, ck = {}, [], {}, ""
    for p, q in zip(parts, arms):
        s, m = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
        for k in s.files: act.setdefault(k, []).append(s[k])
        keys += [f"{e}|{t}" for e, t in zip(m["ep"], m["t"])]
        for k in m.files:
            if k.startswith("ok_"): ok.setdefault(k[3:], []).append(m[k])
        ck = str(m["ckpt"])
    return {k: np.concatenate(v) for k, v in act.items()}, keys, {k: np.concatenate(v) for k, v in ok.items()}, ck

ref, rkeys, _, _ = load("nogate")
real = ref["real"]; thr = (ref["demo"][..., 6].min() + ref["demo"][..., 6].max()) / 2
cl = real[..., 6] > thr; G = cl.all(1) != cl.any(1)
print(f"grasp windows: {G.sum()} / {len(G)}")
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ts = datetime.datetime.now().isoformat(timespec="seconds"); rows = []
for sub, pre in SRC:
    A, keys, OK, ck = load(sub); idx = [keys.index(k) for k in rkeys]
    assert np.allclose(A["real"][idx], real, atol=1e-5), sub
    for arm in [k for k in A if k not in ("real", "demo")]:
        if sub != "nogate" and arm in ("vae", "zhat"): continue
        v = A[arm][idx]; m = G & (OK[arm][idx].astype(bool) if arm in OK else True)
        if m.sum() == 0: continue
        e = np.abs(v[..., :3] - real[..., :3]).mean((1, 2))[m]; gw = ((v[..., 6] > thr) != cl)[m].mean()
        name = arm if pre is None else f"{pre}_{arm}"
        snr = arm.split("_")[1][:-2] if "dB" in arm else ""; k = arm.split("_k")[1] if "_k" in arm else ("4" if arm.startswith("wz") else "")
        print(f"  {name:22s} n={m.sum():3d}  xyz={e.mean():.4f} (se {e.std(ddof=1) / np.sqrt(len(e)):.4f})  grip_wrong={100 * gw:.1f}%")
        for met, val in (("a_real_xyz", e.mean()), ("a_real_xyz_se", e.std(ddof=1) / np.sqrt(len(e))), ("grip_wrong", gw)):
            rows.append([name, k, snr, "CDL-C", "GRASP", met, f"{val:.6f}", int(m.sum()), 0, ck, sha, ts])
with open(CSV, "a", newline="") as f: csv.writer(f).writerows(rows)
print(f"appended {len(rows)} rows to {CSV}")
