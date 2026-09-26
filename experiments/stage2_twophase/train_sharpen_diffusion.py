"""Sharpening diffusion (no GAN): a DDPM x0-denoiser of z_t CONDITIONED on the direct
decode z_direct (the blurry-but-right-cube reconstruction from the frozen direct JSCC).

Why this works where s~-conditioned diffusion didn't: z_direct already carries the cube
in the RIGHT place (direct decode +85%), so it is STRONG conditioning — the diffusion
only has to project the blurry latent onto the sharp manifold (deblur), not decode the
cube location from an incomplete signal. Sampling (SDEdit from z_direct) gives a SHARP,
in-distribution cube; masking keeps the background = z_hat. Pure diffusion sampling, no
adversarial loss.

Reuses RefinementDiffusion as the denoiser: cond = z_direct (z_hat slot), s_tilde = 0
(unused here). pred_x0 = z_direct + delta ; delta learns the sharpening residual.

    python experiments/stage2_twophase/train_sharpen_diffusion.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --direct outputs/direct_jscc/direct.pt \
        --out outputs/sharpen_diff --epochs 12 --tmax 400
"""
import argparse, glob, os, pathlib, sys
import torch, torch.nn as nn

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                                        # noqa: E402
from models import JsccEncoder, FadingChannel, RefinementDiffusion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--direct", required=True, help="frozen direct JSCC checkpoint (direct.pt)")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--tmax", type=int, default=400, help="max DDPM t (sharpening needs only low-mid noise)")
ap.add_argument("--dist_weight", type=float, default=15.0)
a = ap.parse_args()
DEV="cuda:0"; os.makedirs(a.out, exist_ok=True)

# frozen direct JSCC (produces z_direct)
class Dec(nn.Module):
    def __init__(s,Din):
        super().__init__(); s.net=nn.Sequential(nn.Linear(Din,1024),nn.GELU(),
            nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,4*28*28))
    def forward(s,x): return s.net(x).view(-1,4,28,28)
dk=torch.load(a.direct,map_location="cpu"); D_jscc=dk["cfg"]["D_jscc"]; grid=dk["cfg"]["grid"]
enc=JsccEncoder(4,28,28,D_jscc,spatial=True,grid=grid).to(DEV); enc.load_state_dict(dk["enc"]); enc.eval()
dec=Dec(D_jscc).to(DEV); dec.load_state_dict(dk["dec"]); dec.eval()
for p in list(enc.parameters())+list(dec.parameters()): p.requires_grad_(False)
ch=FadingChannel(a.snr, channel_type="awgn")

ZT=[]; DIST=[]
for f in sorted(glob.glob(os.path.join(a.pre,"shard*.pt"))):
    d=torch.load(f); ZT.append(d["ZT"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT=torch.cat(ZT).reshape(-1,4,28,28); DIST=torch.cat(DIST); N=len(ZT)
print(f"[data] {N} frames  D_jscc={D_jscc} grid={grid} snr={a.snr} tmax={a.tmax}", flush=True)

CONFIG["snr_db"]=a.snr; CONFIG["channel_type"]="awgn"
rd=RefinementDiffusion(D_jscc,4,28,28,CONFIG.get("refine_d_model",256),CONFIG.get("refine_n_heads",8),
                       CONFIG.get("refine_n_layers",4),sig_spatial=True,sig_grid=grid).to(DEV)
opt=torch.optim.AdamW(rd.parameters(),lr=a.lr)
Tmax=min(a.tmax,len(rd.alphas_cumprod)-1)
g=torch.Generator().manual_seed(0); perm=torch.randperm(N,generator=g); tr=perm[int(N*0.1):]

for e in range(1,a.epochs+1):
    rd.train(); order=tr[torch.randperm(len(tr))]; L=0.0; nb=0
    for i in range(0,len(order),a.bs):
        sel=order[i:i+a.bs]; z=ZT[sel].to(DEV); dm=DIST[sel].to(DEV); B=len(z)
        with torch.no_grad():
            _,_,s=enc(z,sample=False); st=ch(s); z_direct=dec(st)      # blurry-but-right-cube base
        zero_s=torch.zeros(B,D_jscc,device=DEV)
        t=torch.randint(1,Tmax+1,(B,),device=DEV)
        ab=rd.alphas_cumprod[t].view(B,1,1,1).float()
        x_t=ab.sqrt()*z+(1.0-ab).sqrt()*torch.randn_like(z)           # noise TARGET z_t
        pred=rd._denoise_x0(x_t,t,z_direct,zero_s)                    # condition on z_direct
        ds=((pred-z)**2).mean(dim=(1,2,3)); fw=1.0+a.dist_weight*dm
        loss=(fw*ds).sum()/fw.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(rd.parameters(),5.0); opt.step()
        L+=loss.item(); nb+=1
    print(f"ep{e:>2}/{a.epochs}  train {L/nb:.4f}", flush=True)
    torch.save({"rd":rd.state_dict(),"cfg":{"D_jscc":D_jscc,"grid":grid,"snr":a.snr},
                "direct":a.direct}, os.path.join(a.out,"sharpen.pt"))
print("DONE", flush=True)
