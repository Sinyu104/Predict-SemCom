"""Evaluate a conv-WZ codec on the SAME held-out latent split / CDL channel / metric as baselines/dvst/eval.py.

Mirrors the deployed wz2cam path (server/semcom_server.py): m from z_t vs the true previous latent, encode z_t*m,
send the first k channels through the channel, fill the rest with the prior mean mu_p(z_hat), decode, then
composite z~ = g*rec + (1-g)*z_hat.  Reports latent MSE of z~ (and of z_hat alone, the prediction baseline).
Rate: frac16 = k/16 (same axis as DVST's frac16; neither side counts its side info: WZ's mask m, DVST's k-map).

  python baselines/dvst/eval_wz.py --ckpt outputs/wz2cam_retrain/F_cdl_10_20/masked_alloc.pt --snr 10 12.5 15 17.5 20 --k 2 4 8
Appends to results/wz_vs_dvst.csv (same columns as dvst.csv; `beta` = WZ beta).
"""
import argparse, csv, datetime, os, pathlib, subprocess, sys, torch
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "experiments" / "stage2_twophase"))
from models import FadingChannel      # noqa: E402
from conv_codec import ConvWZCodec    # noqa: E402
from data import split                # noqa: E402
import glob

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--pre", nargs="+", default=["outputs/rollout_disturbed_2cam", "outputs/rollout_calm_2cam"])
ap.add_argument("--snr", type=float, nargs="+", required=True)
ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--channel", default="cdl")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--csv", default=str(REPO / "results" / "wz_vs_dvst.csv"))
a = ap.parse_args(); torch.manual_seed(a.seed); DEV = a.device

ck = torch.load(a.ckpt, map_location=DEV); c = ck["cfg"]; assert c["arch"] == "conv_wz"
net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"]).to(DEV); net.load_state_dict(ck["net"]); net.eval()
ZT, ZH, ZP, DIST = [], [], [], []
for f in sorted(f for d in a.pre for f in glob.glob(os.path.join(d, "shard*.pt"))):
    d = torch.load(f, map_location="cpu"); ZT.append(d["ZT"].float()); ZH.append(d["ZH"].float()); ZP.append(d["ZPREV"].float())
ZT, ZH, ZP = torch.cat(ZT), torch.cat(ZH), torch.cat(ZP); _, va = split(len(ZT))      # identical split to the DVST trainer
ch = FadingChannel(20.0, a.channel, {"model": "C", "delay_spread": 100e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0})
sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ts = datetime.datetime.now().isoformat(timespec="seconds"); os.makedirs(os.path.dirname(a.csv), exist_ok=True)
new = not os.path.exists(a.csv); f = open(a.csv, "a", newline=""); wr = csv.writer(f)
if new: wr.writerow("beta,k,snr_db,channel,split,metric,value,seed,ckpt,git_sha,timestamp".split(","))
with torch.no_grad():
    for snr in a.snr:
        ch.snr_db = snr
        for k in a.k:
            ez = eh = 0.0
            for i in range(0, len(va), 128):
                z = ZT[va[i:i + 128]].to(DEV); zh = ZH[va[i:i + 128]].to(DEV); zp = ZP[va[i:i + 128]].to(DEV)
                m = (((z - zp) ** 2).mean(1, keepdim=True) > c["thr"]).float()
                mu_e, _ = net.encode(z * m); mu_p, _ = net.prior(zh)
                s = ch(mu_e[:, :k].flatten(1)).view(len(z), k, *mu_e.shape[2:])
                rec = net.decode(torch.cat([s, mu_p[:, k:]], 1))
                g = m * (((rec - zh) ** 2).mean(1, keepdim=True) > c["recv_thr"]).float()
                zt = g * rec + (1 - g) * zh
                ez += ((zt - z) ** 2).mean((1, 2, 3)).sum().item(); eh += ((zh - z) ** 2).mean((1, 2, 3)).sum().item()
            n = len(va)
            for met, v in (("latent_mse", ez / n), ("zhat_mse", eh / n), ("frac16", k / 16)):
                wr.writerow([c["beta"], k, snr, a.channel, "ALL", met, f"{v:.6f}", a.seed, a.ckpt, sha, ts])
            print(f"snr {snr:>4g} k={k:<2d} frac16 {k/16:.3f}  mse(z~) {ez/n:.4f}   zhat-only {eh/n:.4f}", flush=True)
f.close()
