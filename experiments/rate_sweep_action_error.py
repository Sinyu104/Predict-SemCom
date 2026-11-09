"""8 dB rate sweep k = 2, 4, 8, 16: offline PI0Fast action error vs a* (true images) on the same 200 held-out windows, for
WZ-SC (K_nogate_snr), DVST rate-matched (eta scaled so frac16 ~ k/16) and SSCC rate-matched (H.264 + NR MCS11, other agent's
outputs/cmp_pose_unf2/sscc_match/k<k>, received windows only).  Splits ALL and GRASP (a*'s chunk has a gripper open<->close switch).
Appends rows to results/wz_offline_pi0fast.csv (arm wz/dvst/sscc_match, k, snr 8): a_real_xyz, a_real_xyz_se, grip_wrong, and for
DVST frac16, for SSCC recv_frac.

    python experiments/rate_sweep_action_error.py
"""
import csv, datetime, glob, subprocess
import numpy as np
D = "outputs/cmp_pose_unf2"; CSV = "results/wz_offline_pi0fast.csv"; KS = [2, 4, 8, 16]

def load(pairs):
    """pairs: [(scores_actions.npz, arms.npz)] -> actions dict, ep|t keys, ok flags, extras"""
    act, keys, ok, ex = {}, [], {}, {}
    for p, q in pairs:
        s, m = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
        for k in s.files: act.setdefault(k, []).append(s[k])
        keys += [f"{e}|{t}" for e, t in zip(m["ep"], m["t"])]
        for k in m.files:
            if k.startswith("ok_"): ok.setdefault(k[3:], []).append(m[k])
            if k == "frac16": ex.setdefault(k, []).append(m[k])
        ex["ckpt"] = str(m["ckpt"])
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return cat(act), keys, cat(ok), {k: (np.concatenate(v) if k != "ckpt" else v) for k, v in ex.items()}

ref, rkeys, _, rex = load([(f"{D}/nogate/scores_part{i}_actions.npz", f"{D}/nogate/arms_part{i}.npz") for i in (0, 1)])
real = ref["real"]; thr = (ref["demo"][..., 6].min() + ref["demo"][..., 6].max()) / 2
cl = real[..., 6] > thr; GR = cl.all(1) != cl.any(1)

def stats(v, m):
    e = np.abs(v[..., :3] - real[..., :3]).mean((1, 2))[m]
    return e.mean(), e.std(ddof=1) / np.sqrt(len(e)), ((v[..., 6] > thr) != cl)[m].mean(), int(m.sum())

def aligned(A, keys, arm):
    idx = [keys.index(k) for k in rkeys]; assert np.allclose(A["real"][idx], real, atol=1e-5); return A[arm][idx], idx

k8, k8k, _, k8x = load([(f"{D}/nogate_k8dB/scores_actions.npz", f"{D}/nogate_k8dB/arms.npz")])
res = []                                                                         # (arm, k, split, metrics..., extra, ckpt)
for k in KS:
    if k == 4: v, _ = aligned(ref, rkeys, "wz_8dB"); ck = rex["ckpt"]
    else: v, _ = aligned(k8, k8k, f"wz_8dB_k{k}"); ck = k8x["ckpt"]
    res.append(("wz", k, v, np.ones(len(real), bool), {}, ck))
for k in KS:
    A, keys, _, ex = load([(f"{D}/rate_sweep/dvst_k{k}_actions.npz", f"{D}/rate_sweep/dvst_k{k}.npz")])
    v, idx = aligned(A, keys, "dvst_8dB"); res.append(("dvst", k, v, np.ones(len(real), bool), {"frac16": ex["frac16"][idx].mean()}, ex["ckpt"]))
for k in KS:
    pairs = []
    for i in range(4):
        sc = f"{D}/sscc_match/k{k}/scores_part{i}_actions.npz"
        if not glob.glob(sc): sc = f"{D}/rate_sweep/sscc_k{k}_part{i}_actions.npz"   # k2 shard 0 rescored here (its run was cut off)
        pairs.append((sc, f"{D}/sscc_match/k{k}/arms_part{i}.npz"))
    A, keys, OK, ex = load(pairs)
    v, idx = aligned(A, keys, "sscc_8dB"); ok = OK["sscc_8dB"][idx].astype(bool)
    res.append(("sscc_match", k, v, ok, {"recv_frac": ok.mean()}, ex["ckpt"]))

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ts = datetime.datetime.now().isoformat(timespec="seconds"); rows = []
print(f"8 dB CDL, n={len(real)} windows ({GR.sum()} grasp)\n{'arm':11s}{'k':>3s} | {'ALL xyz':>8s} {'grip%':>6s} | {'GRASP xyz':>9s} {'se':>7s} {'grip%':>6s} {'n':>4s} | extra")
for arm, k, v, ok, ex, ck in res:
    line = f"{arm:11s}{k:3d} |"
    for split, m in (("ALL", ok), ("GRASP", ok & GR)):
        if m.sum() == 0: line += "      (no received window)      |"; continue
        mu, se, gw, n = stats(v, m)
        line += f" {mu:8.4f} {100 * gw:5.1f}% |" if split == "ALL" else f" {mu:9.4f} {se:7.4f} {100 * gw:5.1f}% {n:4d} |"
        for met, val in (("a_real_xyz", mu), ("a_real_xyz_se", se), ("grip_wrong", gw)):
            rows.append([arm, k, 8, "CDL-C", split, met, f"{val:.6f}", n, 0, ck, sha, ts])
    for met, val in ex.items():
        line += f" {met}={val:.3f}"; rows.append([arm, k, 8, "CDL-C", "ALL", met, f"{val:.6f}", len(real), 0, ck, sha, ts])
    print(line)
with open(CSV, "a", newline="") as f: csv.writer(f).writerows(rows)
print(f"appended {len(rows)} rows to {CSV}")
