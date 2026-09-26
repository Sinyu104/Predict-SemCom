"""Direct JSCC autoencoder + ADVERSARIAL loss, so the reconstruction is SHARP and
IN-DISTRIBUTION (on the real-latent manifold), not the blurry L2 mean.

Rationale (from the masked-direct result): masking already keeps the background
in-distribution (from z_hat); s~ carries the cube (direct +85%); the only remaining
gap is that the direct-decoded cube is BLURRY (regression->mean), so the VLA prefers
z_hat's sharp-wrong cube. A latent-space discriminator pushes z~ onto the manifold of
real z_t -> the VAE then decodes it to a SHARP, in-distribution image. Reconstruction
loss keeps the cube correct; adversarial loss makes it realistic. LSGAN (stable).

Checkpoint format matches train_direct_jscc.py (enc/dec/cfg) so decode_arms_direct.py
consumes it unchanged for the masked-direct VLA eval.

    python experiments/stage2_twophase/train_direct_gan.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/direct_gan \
        --epochs 25 --lam_adv 0.1
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
ap.add_argument("--grid", type=int, default=8)
ap.add_argument("--dist_weight", type=float, default=15.0)
ap.add_argument("--lam_adv", type=float, default=0.1, help="weight on the adversarial (sharpness) loss")
a = ap.parse_args()
DEV="cuda:0"; os.makedirs(a.out, exist_ok=True)

ZT=[]; DIST=[]
for f in sorted(glob.glob(os.path.join(a.pre,"shard*.pt"))):
    d=torch.load(f); ZT.append(d["ZT"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT=torch.cat(ZT).reshape(-1,4,28,28); DIST=torch.cat(DIST); N=len(ZT)
print(f"[data] {N} frames disturbed={int(DIST.sum())}  D_jscc={a.D_jscc} lam_adv={a.lam_adv}", flush=True)

class Dec(nn.Module):
    def __init__(s,Din):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(Din,1024),nn.GELU(),nn.Linear(1024,2048),nn.GELU(),
                            nn.Linear(2048,4*28*28))
    def forward(s,x): return s.net(x).view(-1,4,28,28)

class Disc(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(
            nn.Conv2d(4,64,4,2,1), nn.LeakyReLU(0.2,True),      # 28->14
            nn.Conv2d(64,128,4,2,1), nn.LeakyReLU(0.2,True),    # 14->7
            nn.Conv2d(128,128,3,1,1), nn.LeakyReLU(0.2,True),
            nn.Flatten(), nn.Linear(128*7*7,1))
    def forward(s,x): return s.net(x)

enc=JsccEncoder(4,28,28,a.D_jscc,spatial=True,grid=a.grid).to(DEV)
dec=Dec(a.D_jscc).to(DEV); disc=Disc().to(DEV)
ch=FadingChannel(a.snr, channel_type="awgn")
optG=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=a.lr,betas=(0.5,0.9))
optD=torch.optim.AdamW(disc.parameters(),lr=a.lr,betas=(0.5,0.9))
g=torch.Generator().manual_seed(0); perm=torch.randperm(N,generator=g); tr=perm[int(N*0.1):]

for e in range(1,a.epochs+1):
    enc.train();dec.train();disc.train()
    order=tr[torch.randperm(len(tr))]; Rc=Ad=Dl=0.0; nb=0
    for i in range(0,len(order),a.bs):
        sel=order[i:i+a.bs]; z=ZT[sel].to(DEV); dm=DIST[sel].to(DEV)
        _,_,s=enc(z,sample=True); st=ch(s); zt=dec(st)
        # ---- D step (LSGAN): real z_t -> 1, fake z~ -> 0 ----
        d_real=disc(z); d_fake=disc(zt.detach())
        Ld=0.5*((d_real-1)**2).mean()+0.5*(d_fake**2).mean()
        optD.zero_grad(); Ld.backward(); optD.step()
        # ---- G step: recon (cube correct) + adv (sharp/in-distribution) ----
        ds=((zt-z)**2).mean(dim=(1,2,3)); fw=1.0+a.dist_weight*dm
        rec=(fw*ds).sum()/fw.sum()
        adv=((disc(zt)-1)**2).mean()
        Lg=rec+a.lam_adv*adv
        optG.zero_grad(); Lg.backward(); optG.step()
        Rc+=rec.item(); Ad+=adv.item(); Dl+=Ld.item(); nb+=1
    print(f"ep{e:>2}/{a.epochs}  recon {Rc/nb:.4f}  adv {Ad/nb:.3f}  D {Dl/nb:.3f}", flush=True)
    torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),
                "cfg":{"D_jscc":a.D_jscc,"grid":a.grid,"snr":a.snr}},
               os.path.join(a.out,"direct.pt"))
print("DONE", flush=True)
