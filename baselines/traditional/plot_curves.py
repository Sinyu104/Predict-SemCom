"""
Plot rate-distortion (PSNR vs bpp) and channel (PSNR vs SNR) curves from one
or more result CSVs produced by run_baseline.py.

    python -m baselines.traditional.plot_curves \
        --csv rd_cdlC.csv rd_rayleigh.csv --mode rd  --out rd.png
    python -m baselines.traditional.plot_curves \
        --csv snr_sweep.csv --mode snr --out snr.png
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--mode", choices=["rd", "snr"], default="rd",
                   help="rd = PSNR vs bpp; snr = PSNR vs SNR.")
    p.add_argument("--out", default="curve.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4.5))
    for path in args.csv:
        rows = _read(path)
        # group by a stable label (channel + snr for rd; channel + qp/crf for snr)
        groups = defaultdict(list)
        for r in rows:
            if args.mode == "rd":
                label = f"{r['channel']}@{float(r['snr_db']):g}dB ({os.path.basename(path)})"
                x = float(r["bpp"]);
            else:
                q = r["qp"] or f"crf{r['crf']}"
                label = f"{r['channel']} qp={q} ({os.path.basename(path)})"
                x = float(r["snr_db"])
            groups[label].append((x, float(r["psnr_db"])))
        for label, pts in groups.items():
            pts.sort()
            xs = [a for a, _ in pts]; ys = [b for _, b in pts]
            plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("bits per pixel" if args.mode == "rd" else "SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate-distortion" if args.mode == "rd" else "PSNR vs SNR (cliff effect)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"[plot] saved {args.out}")


if __name__ == "__main__":
    main()
