"""Offline SSCC outage rate on the 200 held-out windows (outputs/cmp_pose_unf2/sscc*, see baselines/traditional/decode_arms_pose.py).
lost   = the target frame of either camera failed CRC (ok_<arm> False)
outage = lost, OR it decoded to a blank picture (either camera's image std < 1: no reference frame to build on, e.g. the keyframe
         did not fit the transport block or an earlier GOP frame was lost) -- the policy gets no usable image.
Appends rows (split ALL, metrics lost_frac / outage_frac) to results/wz_offline_pi0fast.csv.

    python experiments/sscc_outage_rate.py
"""
import csv, datetime, glob, subprocess
import numpy as np
D = "outputs/cmp_pose_unf2"; CSV = "results/wz_offline_pi0fast.csv"
CFG = [("sscc800", "", f"{D}/sscc"), ("sscc_match", 16, f"{D}/sscc_match/k16"), ("sscc_match", 8, f"{D}/sscc_match/k8"),
       ("sscc_match", 4, f"{D}/sscc_match/k4"), ("sscc_match", 2, f"{D}/sscc_match/k2")]
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ts = datetime.datetime.now().isoformat(timespec="seconds"); rows = []
for arm, k, d in CFG:
    P = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{d}/arms_part*.npz"))]; ck = str(P[0]["ckpt"])
    for s in P[0]["snr_db"]:
        a = f"sscc_{s:g}dB"; ok = np.concatenate([p["ok_" + a] for p in P]).astype(bool)
        blank = np.zeros_like(ok)
        if "img_" + a in P[0].files:
            blank = np.concatenate([[im.std() < 1 or w.std() < 1 for im, w in zip(p["img_" + a], p["wimg_" + a])] for p in P])
        lost, out = (~ok).mean(), (~ok | blank).mean()
        print(f"{arm:10s} k={k!s:3s} {s:4g} dB  lost {100 * lost:5.1f}%  outage {100 * out:5.1f}%")
        for met, v in (("lost_frac", lost), ("outage_frac", out)):
            rows.append([arm, k, f"{s:g}", "CDL-C", "ALL", met, f"{v:.6f}", len(ok), 0, ck, sha, ts])
with open(CSV, "a", newline="") as f: csv.writer(f).writerows(rows)
print(f"appended {len(rows)} rows to {CSV}")
