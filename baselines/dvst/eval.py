"""Evaluate a trained DVST checkpoint on the held-out latent split; append stamped rows to results/dvst.csv.

Rows follow the repo convention: beta,k,snr_db,channel,split,metric,value,seed,ckpt,git_sha,timestamp
(`beta` holds DVST's lambda, `k` holds the mean sent reals per cell so the rate axis stays comparable
with the WZ codec's k-of-16 channels).  Metrics: latent_mse, cbr (reals/source real), frac16 (fraction
of the WZ codec's full 16-channel rate, k_total / (16*H*W)).

  python baselines/dvst/eval.py --ckpt outputs/dvst_lam0.1/dvst.pt --pre outputs/rollout_disturbed_clean_s2_moe \
      --snr 0 5 10 15 20 --channel awgn
"""
import argparse, csv, datetime, os, pathlib, subprocess, sys, torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from model import DVST, MaskedChannel   # noqa: E402
from data import load_cache, split       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--pre", required=True, nargs="+")
ap.add_argument("--snr", type=float, nargs="+", default=[0, 5, 10, 15, 20])
ap.add_argument("--channel", default=None, help="default: the channel the ckpt was trained on")
ap.add_argument("--alloc_snr", type=float, default=None, help="hold the rate fixed: allocate with this SNR's capacity at every channel SNR (writes results/dvst_fixedrate.csv)")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default=str(REPO / "results" / "dvst.csv"))
a = ap.parse_args()
torch.manual_seed(a.seed); DEV = a.device

ck = torch.load(a.ckpt, map_location=DEV); c = ck["cfg"]
net = DVST(c["C_vae"], c["cy"], c["cs"], c["w"], c["nb"], eta_prime=c["eta_prime"]).to(DEV)
net.load_state_dict(ck["net"]); net.eval(); net.alloc_snr = a.alloc_snr
if a.alloc_snr is not None and a.csv.endswith("dvst.csv"):
    a.csv = a.csv.replace("dvst.csv", "dvst_fixedrate.csv")
channel = a.channel or c["channel"]
ch = MaskedChannel(20.0, channel).to(DEV)
ZT, ZP, DIST = load_cache(a.pre); _, va = split(len(ZT))
H, W = ZT.shape[2:]
sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ts = datetime.datetime.now().isoformat(timespec="seconds")
os.makedirs(os.path.dirname(a.csv), exist_ok=True)
new = not os.path.exists(a.csv)
f = open(a.csv, "a", newline=""); wr = csv.writer(f)
if new:
    wr.writerow("beta,k,snr_db,channel,split,metric,value,seed,ckpt,git_sha,timestamp".split(","))

with torch.no_grad():
    for snr in a.snr:
        ch.snr_db = snr
        err, kk = [], []
        for i in range(0, len(va), 256):
            sel = va[i:i + 256]; z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV)
            rz, ry = net.intra_reference(zp)
            o = net(z, rz, ry, ch, snr)
            err.append(((o["z_hat"] - z) ** 2).mean(dim=(1, 2, 3)).cpu()); kk.append(o["k_q"].sum(dim=(1, 2, 3)).cpu())
        err = torch.cat(err); kk = torch.cat(kk); dm = DIST[va]
        for sp, m in (("ALL", dm >= 0), ("CALM", dm < 0.5), ("DISTURBED", dm > 0.5)):
            kbar = kk[m].mean().item() / (H * W)
            for met, v in (("latent_mse", err[m].mean().item()), ("cbr", kk[m].mean().item() / ZT[0].numel()),
                           ("frac16", kk[m].mean().item() / (16 * H * W))):
                wr.writerow([c["lam"], f"{kbar:.3f}", snr, channel, sp, met, f"{v:.6f}", a.seed, a.ckpt, sha, ts])
            print(f"snr {snr:>4g} {sp:<9} mse {err[m].mean():.4f}  k/cell {kbar:.2f}  frac16 {kk[m].mean().item()/(16*H*W):.3f}", flush=True)
f.close(); print("appended ->", a.csv)
