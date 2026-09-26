"""Train + save a DIRECT JSCC autoencoder: encode z_t -> channel@20dB -> decode z_t.
(The 'direct' mode that recovered the cube to +85% in test_residual_coding.py — s~ DOES
carry the cube.) Used by decode_arms_direct.py for MASKED-DIRECT reconstruction:
z~ = mask * dec(s~) + (1-mask) * z_hat  (cube from s~, background from z_hat).

    python experiments/stage2_twophase/train_direct_jscc.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/direct_jscc --epochs 20
"""
import argparse, glob, os, pathlib, sys
import torch, torch.nn as nn

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from models import JsccEncoder, FadingChannel    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--D_jscc", type=int, default=1024)
ap.add_argument("--grid", type=int, default=8)
ap.add_argument("--dist_weight", type=float, default=15.0)
a = ap.parse_args()
DEV="cuda:0"; os.makedirs(a.out, exist_ok=True)

ZT=[]; DIST=[]
for f in sorted(glob.glob(os.path.join(a.pre,"shard*.pt"))):
    d=torch.load(f); ZT.append(d["ZT"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT=torch.cat(ZT).reshape(-1,4,28,28); DIST=torch.cat(DIST); N=len(ZT)
print(f"[data] {N} frames disturbed={int(DIST.sum())}  D_jscc={a.D_jscc} grid={a.grid} snr={a.snr}", flush=True)

class Dec(nn.Module):
    def __init__(s,Din):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(Din,1024),nn.GELU(),nn.Linear(1024,2048),nn.GELU(),
                            nn.Linear(2048,4*28*28))
    def forward(s,x): return s.net(x).view(-1,4,28,28)

enc=JsccEncoder(4,28,28,a.D_jscc,spatial=True,grid=a.grid).to(DEV)
dec=Dec(a.D_jscc).to(DEV)
ch=FadingChannel(a.snr, channel_type="awgn")
opt=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=a.lr)
g=torch.Generator().manual_seed(0); perm=torch.randperm(N,generator=g)
tr=perm[int(N*0.1):]

for e in range(1,a.epochs+1):
    enc.train(); dec.train()
    order=tr[torch.randperm(len(tr))]; L=0.0; nb=0
    for i in range(0,len(order),a.bs):
        sel=order[i:i+a.bs]; z=ZT[sel].to(DEV); dm=DIST[sel].to(DEV)
        _,_,s=enc(z,sample=True); st=ch(s); zt=dec(st)
        ds=((zt-z)**2).mean(dim=(1,2,3)); fw=1.0+a.dist_weight*dm
        loss=(fw*ds).sum()/fw.sum()
        opt.zero_grad(); loss.backward(); opt.step(); L+=loss.item(); nb+=1
    print(f"ep{e:>2}/{a.epochs}  train {L/nb:.4f}", flush=True)
    torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),
                "cfg":{"D_jscc":a.D_jscc,"grid":a.grid,"snr":a.snr}},
               os.path.join(a.out,"direct.pt"))
print("DONE", flush=True)
