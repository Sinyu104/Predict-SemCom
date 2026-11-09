"""SSCC above the keyframe floor at 8 dB, NR MCS11 (same MCS as the k=16 rate match): offline PI0Fast action error vs a* on the same
200 held-out windows, for H.264 4:4:4 at 92.25 (= sscc_match/k16), 184.5, 400, 800 kbps.  Splits ALL and GRASP; usable windows only
(target frame of both cameras decoded and not blank).  rho = 2 cams * REs / (3*224*224*2), REs = (TB bytes + 4)*8 / (Qm*R = 1.4766).
Appends rows (arm sscc_mcs11, k = kbps) to results/wz_offline_pi0fast.csv.

    python experiments/sscc_rate_action_error.py
"""
import csv, datetime, glob, subprocess
import numpy as np
D = "outputs/cmp_pose_unf2"; CSV = "results/wz_offline_pi0fast.csv"
RUNS = [(92.25, f"{D}/sscc_match/k16"), (184.5, f"{D}/sscc_rate/kb184.5"), (400, f"{D}/sscc_rate/kb400"), (800, f"{D}/sscc_rate/kb800")]

def load(d):
    S = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{d}/scores_part*_actions.npz"))]
    M = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{d}/arms_part*.npz"))]
    assert len(S) == len(M) == 4, d
    cat = lambda k, X: np.concatenate([x[k] for x in X])
    a = "sscc_8dB"; blank = np.concatenate([[i.std() < 1 or w.std() < 1 for i, w in zip(m["img_" + a], m["wimg_" + a])] for m in M])
    keys = [f"{e}|{t}" for m in M for e, t in zip(m["ep"], m["t"])]
    return cat(a, S), cat("real", S), cat("demo", S), keys, cat("ok_" + a, M).astype(bool) & ~blank, int(M[0]["tb_bytes"]), str(M[0]["ckpt"])

ref = [np.load(f"{D}/nogate/arms_part{i}.npz", allow_pickle=True) for i in (0, 1)]
rkeys = [f"{e}|{t}" for m in ref for e, t in zip(m["ep"], m["t"])]
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ts = datetime.datetime.now().isoformat(timespec="seconds"); rows = []
print(f"{'kbps':>7s} {'rho':>6s} {'usable':>7s} | {'ALL xyz':>8s} {'grip%':>6s} | {'GRASP xyz':>9s} {'se':>7s} {'grip%':>6s} {'n':>3s}")
for kb, d in RUNS:
    v, real, demo, keys, use, tb, ck = load(d); idx = [keys.index(k) for k in rkeys]
    v, real, demo, use = v[idx], real[idx], demo[idx], use[idx]
    thr = (demo[..., 6].min() + demo[..., 6].max()) / 2; cl = real[..., 6] > thr; G = cl.all(1) != cl.any(1)
    rho = (tb + 4) * 8 / 1.4766 / (3 * 224 * 224)
    e = np.abs(v[..., :3] - real[..., :3]).mean((1, 2)); gw = (v[..., 6] > thr) != cl
    line = f"{kb:7g} {rho:6.3f} {use.mean():7.1%} |"
    for split, m in (("ALL", use), ("GRASP", use & G)):
        mu, se, g = e[m].mean(), e[m].std(ddof=1) / np.sqrt(m.sum()), gw[m].mean()
        line += f" {mu:8.4f} {100 * g:5.1f}% |" if split == "ALL" else f" {mu:9.4f} {se:7.4f} {100 * g:5.1f}% {m.sum():3d}"
        for met, val in (("a_real_xyz", mu), ("a_real_xyz_se", se), ("grip_wrong", g)):
            rows.append(["sscc_mcs11", f"{kb:g}", 8, "CDL-C", split, met, f"{val:.6f}", int(m.sum()), 0, ck, sha, ts])
    for met, val in (("rho", rho), ("outage_frac", 1 - use.mean())):
        rows.append(["sscc_mcs11", f"{kb:g}", 8, "CDL-C", "ALL", met, f"{val:.6f}", len(use), 0, ck, sha, ts])
    print(line)
with open(CSV, "a", newline="") as f: csv.writer(f).writerows(rows)
print(f"appended {len(rows)} rows to {CSV}")
