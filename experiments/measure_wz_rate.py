"""Wyner-Ziv rate of a trained conv WZ codec: KL(q(s|z_t) || p(s|z_hat)) in bits per frame, on the codec's own validation split.

Reports, per k (code channels sent): all-cell rate (what the trainer's `rate` sums) and mask-cell rate, vs the analog channel uses.
    python experiments/measure_wz_rate.py --ckpt outputs/wz2cam_retrain/D_b1e-5_dense/masked_alloc.pt --pre outputs/rollout_disturbed_2cam outputs/rollout_calm_2cam
"""
import argparse, glob, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage2_twophase"))
from conv_codec import ConvWZCodec

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--pre", nargs="+", required=True)
ap.add_argument("--out", default=None); ap.add_argument("--dev", default="cuda:0")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu"); c = ck["cfg"]
net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"]).to(a.dev).eval(); net.load_state_dict(ck["net"])
sig_n = 1.0 / 10 ** (c["snr"] / 10)
ZT, ZP, ZH = [], [], []
for f in sorted(f for d in a.pre for f in glob.glob(os.path.join(d, "shard*.pt"))):
    d = torch.load(f); ZT.append(d["ZT"].float()); ZP.append(d["ZPREV"].float()); ZH.append(d["ZH"].float())
v = lambda x: torch.cat(x).reshape(-1, 4, 28, 56)
ZT, ZP, ZH = v(ZT), v(ZP), v(ZH); N = len(ZT)
perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)); va = perm[:int(N * 0.1)]     # same split as the trainer
kl_all = torch.zeros(c["conv_dch"]); kl_msk = torch.zeros(c["conv_dch"]); nm = 0.0
with torch.no_grad():
    for i in range(0, len(va), 128):
        s = va[i:i + 128]; z, zp, h = ZT[s].to(a.dev), ZP[s].to(a.dev), ZH[s].to(a.dev)
        m = (((z - zp) ** 2).mean(1, keepdim=True) > c["thr"]).float()
        mu_e, lv_e = net.encode(z * m); mu_p, lv_p = net.prior(h)
        kl = net.kl_elem(mu_e, lv_e, mu_p, lv_p, sig_n)                     # (B,Dch,H,W) nats
        kl_all += kl.sum((0, 2, 3)).cpu(); kl_msk += (kl * m).sum((0, 2, 3)).cpu(); nm += m.sum().item()
nf = len(va); B = 1 / math.log(2); cells = 28 * 56
res = {"ckpt": a.ckpt, "beta": c["beta"], "n_val_frames": nf, "mask_cells_per_frame": nm / nf, "bits_per_frame": {}}
print(f"{a.ckpt}  beta={c['beta']}  val frames {nf}  mask cells/frame {nm / nf:.0f}/{cells}")
print(f"{'k':>3} {'analog reals':>13} {'WZ rate all cells (bits)':>25} {'WZ rate mask cells (bits)':>26} {'bits/real':>10}")
for k in (1, 2, 4, 8, 16):
    ra = kl_all[:k].sum().item() / nf * B; rm = kl_msk[:k].sum().item() / nf * B; reals = k * cells
    res["bits_per_frame"][k] = {"analog_reals": reals, "all_cells": ra, "mask_cells": rm}
    print(f"{k:>3} {reals:>13} {ra:>25.0f} {rm:>26.0f} {ra / reals:>10.3f}")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
