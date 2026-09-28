"""Change-masked allocation JSCC: concentrate the WHOLE code budget on the changed
(cube) cells so the cube gets over-capacity and reconstructs SHARP.

Transmitter (WZ-clean, uses z_{t-1}, NOT z_hat):
  m     = per-cell (||z_t - z_{t-1}||^2 > thr)     # changed region (the cube)
  z_in  = z_t * m                                  # mask the encoder input to the cube
  s     = enc(z_in) -> channel -> s~               # all D_jscc symbols encode just the cube
Receiver (m is signalled, ~784 bits):
  z~    = m * dec(s~) + (1-m) * z_hat              # cube from channel, background from z_hat

Because the encoder input is ~zero except the cube, the full 1024-symbol budget lands on
the cube (over-capacity, noiseless channel) -> the cube target is fully determined -> L2
reconstructs it SHARP (no ambiguity). Fixes the per-cell-capacity limit that made the
uniform spatial code (4-8 symbols/cube-cell) blurry, and avoids the loss-weighting
collapse (weighting a capacity-limited target -> posterior collapse).

    python experiments/stage2_twophase/train_masked_alloc.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/masked_alloc \
        --thr 0.05 --epochs 25
"""
import argparse, glob, os, pathlib, sys
import torch, torch.nn as nn

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from models import JsccEncoder, FadingChannel    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--D_jscc", type=int, default=1024)
ap.add_argument("--thr", type=float, default=0.15, help="transmitter change-mask threshold (per-cell ||z_t-z_{t-1}||^2)")
ap.add_argument("--recv_thr", type=float, default=0.05,
                help="RECEIVER disagreement threshold: within the transmitted region, only overwrite z_hat "
                     "where ||rec - z_hat||^2 per cell > recv_thr (the cube); keep z_hat elsewhere (the arm). "
                     "Uses z_hat at the receiver (WZ-legal) to isolate the cube the transmitter mask can't.")
ap.add_argument("--dist_weight", type=float, default=15.0)
a = ap.parse_args()
DEV="cuda:0"; os.makedirs(a.out, exist_ok=True)

ZT=[]; ZP=[]; DIST=[]
for f in sorted(glob.glob(os.path.join(a.pre,"shard*.pt"))):
    d=torch.load(f)
    assert "ZPREV" in d, "need ZPREV cache (rollout_disturbed_clean_s2_moe)"
    ZT.append(d["ZT"].float()); ZP.append(d["ZPREV"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
# NOTE: z_hat (ZH) is the receiver prior; loaded separately for the composite
ZH=[]
for f in sorted(glob.glob(os.path.join(a.pre,"shard*.pt"))):
    ZH.append(torch.load(f)["ZH"].float())
ZT=torch.cat(ZT).reshape(-1,4,28,28); ZP=torch.cat(ZP).reshape(-1,4,28,28)
ZH=torch.cat(ZH).reshape(-1,4,28,28); DIST=torch.cat(DIST); N=len(ZT)
print(f"[data] {N} frames disturbed={int(DIST.sum())}  thr={a.thr}", flush=True)

class Dec(nn.Module):
    def __init__(s,Din):
        super().__init__(); s.net=nn.Sequential(nn.Linear(Din,1024),nn.GELU(),
            nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,4*28*28))
    def forward(s,x): return s.net(x).view(-1,4,28,28)

def change_mask(z, zp):
    m = ((z - zp)**2).mean(1, keepdim=True) > a.thr    # (B,1,28,28) changed cells
    return m.float()

enc=JsccEncoder(4,28,28,a.D_jscc,spatial=False).to(DEV)
dec=Dec(a.D_jscc).to(DEV)
ch=FadingChannel(a.snr, channel_type="awgn")
opt=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=a.lr)
g=torch.Generator().manual_seed(0); perm=torch.randperm(N,generator=g); tr=perm[int(N*0.1):]

for e in range(1,a.epochs+1):
    enc.train(); dec.train(); order=tr[torch.randperm(len(tr))]; L=0.0; nb=0; msz=0.0
    for i in range(0,len(order),a.bs):
        sel=order[i:i+a.bs]; z=ZT[sel].to(DEV); zp=ZP[sel].to(DEV); h=ZH[sel].to(DEV); dm=DIST[sel].to(DEV)
        m=change_mask(z,zp)
        z_in=z*m                                        # mask encoder input to the changed region
        _,_,s=enc(z_in,sample=True); st=ch(s); rec=dec(st)
        # RECEIVER-side disagreement composite: within the transmitted region m, only overwrite
        # z_hat where rec disagrees with it (the cube); keep z_hat on the arm (rec ~= z_hat there).
        g = m * (((rec - h)**2).mean(1,keepdim=True) > a.recv_thr).float()
        zt=g*rec+(1-g)*h                                # cube from channel, arm/bg from z_hat
        ds=((zt-z)**2).mean(dim=(1,2,3)); fw=1.0+a.dist_weight*dm
        loss=(fw*ds).sum()/fw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        L+=loss.item(); nb+=1; msz+=m.mean().item()
    print(f"ep{e:>2}/{a.epochs}  loss {L/nb:.4f}  avg mask frac {msz/nb:.3f}", flush=True)
    torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),
                "cfg":{"D_jscc":a.D_jscc,"thr":a.thr,"recv_thr":a.recv_thr,"snr":a.snr,"spatial":False}},
               os.path.join(a.out,"masked_alloc.pt"))
print("DONE", flush=True)
