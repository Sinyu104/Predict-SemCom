"""Calibrate NR TB error rate per (MCS, SNR) at the video frame size -> nr_bler_tb<bytes>.json
    python -m baselines.traditional.calibrate_nr --tb_bytes 1250 --channels awgn rayleigh cdl
"""
import argparse, numpy as np
from .nr_link import NRLink

ap = argparse.ArgumentParser()
ap.add_argument("--tb_bytes", type=int, default=1250)
ap.add_argument("--channels", nargs="+", default=["awgn", "rayleigh", "cdl"])
ap.add_argument("--cdl_model", default="C")
ap.add_argument("--snrs", type=float, nargs="+", default=list(np.arange(-2, 23, 1.0)))
ap.add_argument("--blocks", type=int, default=150)
a = ap.parse_args()
for ch in a.channels:
    link = NRLink(ch, 0.0, tb_bytes=a.tb_bytes, cdl_model=a.cdl_model, seed=0)
    link.calibrate(a.snrs, blocks=a.blocks)
    print("calibrated", ch, "->", link.table_path, flush=True)
