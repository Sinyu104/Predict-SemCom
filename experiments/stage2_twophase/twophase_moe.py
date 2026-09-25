"""Adaptive-rate (MoE-router) Wyner-Ziv Stage-2 trainer.

Trains AdaptiveSemComSystem: a TemporalRouter picks a code LENGTH from
||z_t - z_{t-1}|| (WZ-clean, no prediction at the TX), nested-prefix JSCC sends
that many tokens, and the DisagreementFusionRefiner reconstructs. The routing
level is the disturbance signal the receiver keys on — calm frames send few
tokens (channel quiet -> gate closed -> z̃=z_hat), disturbed frames send many
(channel carries the cube).

    L = distortion(z̃, z_t)  +  beta * rate(KL over sent tokens)  +  lam_len * (E[tokens]/max)
distortion up-weights TRUE disturbed frames (DIST); rate is the WZ conditional KL
charged only on transmitted tokens; the length term keeps codes short unless the
distortion demands otherwise -> the router learns "short on calm, long on disturbed".

    python experiments/stage2_twophase/twophase_moe.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/stage2_moe \
        --snr 20 --channel awgn --beta 0.02 --lam_len 0.5 --dist_weight 15 --epochs 12
"""
import argparse, glob, os, pathlib, sys
import torch, torch.nn as nn

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                    # noqa: E402
from models import AdaptiveSemComSystem      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True, help="cache dir with ZT/ZH/DIST/ZPREV shards")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--channel", default="awgn")
ap.add_argument("--beta", type=float, default=0.02, help="weight on the WZ rate (KL over sent tokens)")
ap.add_argument("--lam_len", type=float, default=0.5, help="weight on expected code length (E[tokens]/max)")
ap.add_argument("--dist_weight", type=float, default=15.0, help="up-weight TRUE disturbed frames in distortion")
ap.add_argument("--warmup_epochs", type=int, default=5,
                help="Force full-length codes (router bypassed, no length penalty) for this many "
                     "epochs so the encoder/refiner learn to USE the channel before the router is "
                     "allowed to shorten codes. Without it the router collapses to the minimum "
                     "length before the channel is ever useful.")
ap.add_argument("--tau0", type=float, default=2.0, help="Gumbel temperature start (after warmup)")
ap.add_argument("--tau1", type=float, default=0.5, help="Gumbel temperature end")
ap.add_argument("--max_tokens", type=int, default=64)
ap.add_argument("--token_dim", type=int, default=16)
ap.add_argument("--levels", type=int, nargs="+", default=[8, 16, 32, 64])
ap.add_argument("--val_frac", type=float, default=0.1)
a = ap.parse_args()

CONFIG["snr_db"] = a.snr; CONFIG["channel_type"] = a.channel
CONFIG["jscc_max_tokens"] = a.max_tokens; CONFIG["jscc_token_dim"] = a.token_dim
CONFIG["jscc_levels"] = a.levels
DEV = "cuda:0"; os.makedirs(a.out, exist_ok=True)
print(f"[cfg] MoE-router adaptive JSCC  levels(tokens)={a.levels} x {a.token_dim} dims  "
      f"max D_jscc={a.max_tokens*a.token_dim}  {a.channel} {a.snr}dB  beta={a.beta} lam_len={a.lam_len}",
      flush=True)

# ── data ──────────────────────────────────────────────────────────────────── #
ZT=[]; ZH=[]; ZP=[]; DIST=[]
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f)
    assert "ZPREV" in d, f"{f} has no ZPREV — rebuild the cache with augment_zprev"
    ZT.append(d["ZT"]); ZH.append(d["ZH"]); ZP.append(d["ZPREV"])
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT=torch.cat(ZT).float().reshape(-1,4,28,28); ZH=torch.cat(ZH).float().reshape(-1,4,28,28)
ZP=torch.cat(ZP).float().reshape(-1,4,28,28); DIST=torch.cat(DIST).float()
N=len(ZT)
g=torch.Generator().manual_seed(0); perm=torch.randperm(N, generator=g)
nval=int(N*a.val_frac); va=perm[:nval]; tr=perm[nval:]
print(f"[data] {N} frames  disturbed={int(DIST.sum())} ({100*DIST.mean():.1f}%)  "
      f"train {len(tr)} val {len(va)}", flush=True)

sysm = AdaptiveSemComSystem(CONFIG).to(DEV)
opt = torch.optim.AdamW(sysm.parameters(), lr=a.lr)
params = list(sysm.parameters())

def run_epoch(idx, train, tau, warmup):
    sysm.train(train)
    order = idx[torch.randperm(len(idx))] if train else idx
    lam_len = 0.0 if warmup else a.lam_len          # no length pressure during warmup
    D=Rt=Ln=G=0.0; nb=0
    for i in range(0, len(order), a.bs):
        sel=order[i:i+a.bs]
        z=ZT[sel].to(DEV); h=ZH[sel].to(DEV); zp=ZP[sel].to(DEV); dm=DIST[sel].to(DEV)
        with torch.set_grad_enabled(train):
            out=sysm(z, h, zp, tau=tau, hard=False, sample=train, force_full=warmup)
            ds=((out["z_tilde"]-z)**2).mean(dim=(1,2,3))
            fw=1.0+a.dist_weight*dm
            d=(fw*ds).sum()/fw.sum()
            r=out["rate"].mean()
            ln=(out["exp_tokens"]/a.max_tokens).mean()
            loss=d + a.beta*r + lam_len*ln
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        D+=d.item(); Rt+=r.item(); Ln+=ln.item(); nb+=1
    return D/nb, Rt/nb, Ln/nb

for e in range(1, a.epochs+1):
    warmup = e <= a.warmup_epochs
    tau=a.tau0 + (a.tau1-a.tau0)*max(0,e-a.warmup_epochs-1)/max(1,a.epochs-a.warmup_epochs-1)
    d,r,ln=run_epoch(tr, True, tau, warmup)
    vd,vr,vln=run_epoch(va, False, tau, warmup)
    tag="[warmup]" if warmup else ""
    print(f"ep{e:>2}/{a.epochs} {tag} tau={tau:.2f}  train d={d:.4f} rate={r:.4f} len={ln:.3f}  "
          f"val d={vd:.4f} rate={vr:.4f} len={vln:.3f}", flush=True)
    torch.save({"system_state": sysm.state_dict(),
                "config": {k: CONFIG[k] for k in ("snr_db","channel_type","jscc_max_tokens",
                                                  "jscc_token_dim","jscc_levels")}},
               os.path.join(a.out, "moe.pt"))

# ── final eval: adaptivity (tokens sent) + reconstruction, split calm/disturbed ── #
sysm.eval()
import numpy as np
tok_c=[]; tok_d=[]; A_c=[]; A_d=[]; C_c=[]; C_d=[]
with torch.no_grad():
    for i in range(0, N, 64):
        sel=torch.arange(i, min(i+64, N))
        z=ZT[sel].to(DEV); h=ZH[sel].to(DEV); zp=ZP[sel].to(DEV); dm=DIST[sel]>0.5
        out=sysm(z, h, zp, hard=True, sample=False)
        toks=out["exp_tokens"].cpu()
        a_=((out["z_tilde"]-z)**2).mean(dim=(1,2,3)).cpu()
        c_=((h-z)**2).mean(dim=(1,2,3)).cpu()
        for k in range(len(sel)):
            (tok_d if dm[k] else tok_c).append(toks[k].item())
            (A_d if dm[k] else A_c).append(a_[k].item()); (C_d if dm[k] else C_c).append(c_[k].item())
def m(x): return float(np.mean(x)) if x else float("nan")
print("\n=== ADAPTIVITY (hard routing) ===")
print(f"  CALM      : tokens sent avg {m(tok_c):5.1f}/{a.max_tokens}   recon A={m(A_c):.4f}  z_hat C={m(C_c):.4f}  A-vs-C={100*(m(C_c)-m(A_c))/m(C_c):+.1f}%")
print(f"  DISTURBED : tokens sent avg {m(tok_d):5.1f}/{a.max_tokens}   recon A={m(A_d):.4f}  z_hat C={m(C_d):.4f}  A-vs-C={100*(m(C_d)-m(A_d))/m(C_d):+.1f}%")
print(f"  avg code length: calm {m(tok_c)*a.token_dim:.0f} symbols  vs  disturbed {m(tok_d)*a.token_dim:.0f} symbols")
print("DONE", flush=True)
