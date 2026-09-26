"""Focused test: does encoding the TEMPORAL RESIDUAL z_t - z_{t-1} make s~ carry the
cube (drive the residual cube error toward 0), vs encoding z_t directly?

Clean JSCC autoencoder (encode -> channel@20dB -> decode), NO diffusion/refiner, to
isolate the CODING question. Three modes, all reconstruct z_t; the difference is what
is encoded and the receiver reference:
    direct        : s=enc(z_t),           z~ = dec(s~)
    resid_zhat    : s=enc(z_t),           z~ = z_hat  + dec(s~)      (current-style)
    resid_temporal: s=enc(z_t - z_prev),  z~ = z_prev + dec(s~)      (NEW — WZ-clean, TX has z_{t-1})
Report cube-region vs background reconstruction error on disturbed frames. If
resid_temporal drives the cube error far below the others, the temporal residual
concentrates capacity on the cube and s~ becomes determining.

    python experiments/stage2_twophase/test_residual_coding.py --pre outputs/rollout_disturbed_clean_s2_moe
"""
import argparse, glob, os, pathlib, sys
import torch, torch.nn as nn, numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                    # noqa: E402
from models import JsccEncoder, FadingChannel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--D_jscc", type=int, default=1024)
ap.add_argument("--grid", type=int, default=8)
a = ap.parse_args()
DEV="cuda:0"

ZT=[];ZH=[];ZP=[];D=[]
for f in sorted(glob.glob(os.path.join(a.pre,"shard*.pt"))):
    d=torch.load(f); ZT.append(d["ZT"].float()); ZH.append(d["ZH"].float())
    ZP.append(d["ZPREV"].float()); D.append(d["DIST"].float())
ZT=torch.cat(ZT).reshape(-1,4,28,28); ZH=torch.cat(ZH).reshape(-1,4,28,28)
ZP=torch.cat(ZP).reshape(-1,4,28,28); D=torch.cat(D)
N=len(ZT); g=torch.Generator().manual_seed(0); perm=torch.randperm(N,generator=g)
nval=int(N*0.1); va=perm[:nval]; tr=perm[nval:]
dm_all = D>0.5
print(f"[data] {N} frames  disturbed={int(D.sum())}", flush=True)

class Dec(nn.Module):
    def __init__(s,Din):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(Din,1024),nn.GELU(),nn.Linear(1024,2048),nn.GELU(),
                            nn.Linear(2048,4*28*28))
    def forward(s,x): return s.net(x).view(-1,4,28,28)

def train_mode(mode):
    enc=JsccEncoder(4,28,28,a.D_jscc,spatial=True,grid=a.grid).to(DEV)
    dec=Dec(a.D_jscc).to(DEV)
    ch=FadingChannel(a.snr, channel_type="awgn")
    opt=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=a.lr)
    def recon(z,h,zp,train):
        src = (z - zp) if mode=="resid_temporal" else z
        enc.train(train)
        _,_,s = enc(src, sample=train); st=ch(s)
        d = dec(st)
        if mode=="direct":         return d
        if mode=="resid_zhat":     return h + d
        return zp + d              # resid_temporal
    for e in range(1,a.epochs+1):
        order=tr[torch.randperm(len(tr))]
        for i in range(0,len(order),a.bs):
            sel=order[i:i+a.bs]; z=ZT[sel].to(DEV);h=ZH[sel].to(DEV);zp=ZP[sel].to(DEV)
            zt=recon(z,h,zp,True); loss=((zt-z)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    # eval on disturbed val frames: cube-region vs background reconstruction
    enc.eval(); dec.eval()
    idx=va[dm_all[va]]
    cube_e=bg_e=czhat=bzhat=0.0; n=0
    with torch.no_grad():
        for i in range(0,len(idx),128):
            sel=idx[i:i+128]; z=ZT[sel].to(DEV);h=ZH[sel].to(DEV);zp=ZP[sel].to(DEV)
            zt=recon(z,h,zp,False)
            dev=((z-h)**2).mean(1).reshape(len(sel),-1)          # cube localizer (vs prediction)
            top=dev.topk(16,1).indices; m=torch.zeros_like(dev,dtype=torch.bool); m.scatter_(1,top,True)
            ea=((zt-z)**2).mean(1).reshape(len(sel),-1); ez=((h-z)**2).mean(1).reshape(len(sel),-1)
            cube_e+=ea[m].sum().item(); bg_e+=ea[~m].sum().item()
            czhat+=ez[m].sum().item(); bzhat+=ez[~m].sum().item(); n+=len(sel)
    return cube_e/(n*16), bg_e/(n*768), czhat/(n*16), bzhat/(n*768)

print(f"{'mode':16}{'cube recon':>12}{'cube zhat':>11}{'bg recon':>10}{'bg zhat':>9}", flush=True)
for mode in ["direct","resid_zhat","resid_temporal"]:
    ce,be,cz,bz=train_mode(mode)
    print(f"{mode:16}{ce:>12.4f}{cz:>11.4f}{be:>10.4f}{bz:>9.4f}   "
          f"cube {100*(cz-ce)/cz:+.0f}% vs zhat", flush=True)
print("DONE", flush=True)
