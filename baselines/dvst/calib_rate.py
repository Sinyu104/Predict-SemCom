"""Replay logged online latents (server --log_dir npz) through DVST with GOP chaining to find the eta' scale giving a target mean frac16."""
import argparse, glob, os, pathlib, re, sys, numpy as np, torch
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from model import DVST, MaskedChannel
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--log", required=True)
ap.add_argument("--alloc_snr", type=float, default=20.0); ap.add_argument("--snr", type=float, default=16.0); ap.add_argument("--gop", type=int, default=5)
ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6]); a = ap.parse_args()
DEV = "cuda"; ck = torch.load(a.ckpt, map_location=DEV); c = ck["cfg"]
net = DVST(c["C_vae"], c["cy"], c["cs"], c["w"], c["nb"], eta_prime=c["eta_prime"]).to(DEV); net.load_state_dict(ck["net"]); net.eval(); net.entropy.ste = True; net.alloc_snr = a.alloc_snr
ch = MaskedChannel(a.snr, "cdl", {"model": "C", "delay_spread": 100e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}).to(DEV)
fs = sorted(glob.glob(os.path.join(a.log, "*.npz")), key=lambda f: tuple(int(x) for x in re.findall(r"\d+", os.path.basename(f))))
eps = {}
for f in fs: eps.setdefault(os.path.basename(f).split("_")[0], []).append(torch.from_numpy(np.load(f)["lat_zt"]).float().unsqueeze(0))
eta0 = net.eta_prime
for s in a.scales:
    net.eta_prime = eta0 * s; F, M = [], []
    with torch.no_grad():
        for zs in eps.values():
            tx = rx = None
            for n, z in enumerate(zs):
                z = z.to(DEV)
                if n % a.gop == 0: tx = rx = None
                ct, dt = net.contexts(*tx) if tx is not None else (None, None); cr, dr = net.contexts(*rx) if rx is not None else (None, None)
                e = net.encode(z, ct, dt, a.snr); st = e["s"] * e["keep"]
                ys, zsim = net.decode(st, e["idx"], ct, dt); yh, zh = net.decode(ch(st, e["keep"]), e["idx"], cr, dr)
                tx, rx = (zsim, ys), (zh, yh); F.append(e["k_q"].sum().item() / (16 * 28 * 56)); M.append(((zh - z) ** 2).mean().item())
    print(f"scale {s:.2f}  frac16 {np.mean(F):.3f}  latent mse {np.mean(M):.4f}  ({len(F)} requests)", flush=True)
