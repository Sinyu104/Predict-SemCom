"""Masked-diffusion Stage-2 trainer.

Trains RefinementDiffusion as a proper multi-level DDPM x0-denoiser of z_t,
conditioned on (z_hat, s~). Sampling (not 1-step MSE regression) is what avoids the
L2 blur; masked sampling at inference (masked_sample) confines the sharp generation
to the cube region and leaves the background = z_hat, so no over-correction.

Training is standard DDPM (no mask): sample t~U[1,Tmax], x_t = sqrt(ab)·z_t +
sqrt(1-ab)·eps, predict x0=z_t. On the cube cells z_hat is wrong, so minimizing the
loss there forces the denoiser to use s~. Disturbed frames up-weighted (dist_weight).
Fixed D_jscc (no router) to isolate the sampling+masking question.

    python experiments/stage2_twophase/twophase_masked.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/stage2_masked \
        --snr 20 --channel awgn --D_jscc 1024 --grid 8 --tmax 600 --epochs 16
"""
import argparse, glob, os, pathlib, sys
import torch, torch.nn as nn

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                    # noqa: E402
from models import SemComSystem              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=16)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--channel", default="awgn")
ap.add_argument("--D_jscc", type=int, default=1024)
ap.add_argument("--grid", type=int, default=8, help="spatial JSCC grid (n_sig = grid^2)")
ap.add_argument("--tmax", type=int, default=999, help="max DDPM timestep sampled in training")
ap.add_argument("--hi_noise_frac", type=float, default=0.6,
                help="fraction of the batch forced to HIGH noise [tmax/2, tmax], where x_t can't "
                     "leak z_t so the model MUST decode s~. The rest is uniform [1, tmax].")
ap.add_argument("--sdrop", type=float, default=0.15,
                help="prob. of dropping s~ (set to 0) during training -> classifier-free guidance, "
                     "so the model learns what s~ ADDS and can be guided to rely on it at sampling.")
ap.add_argument("--dist_weight", type=float, default=15.0)
ap.add_argument("--val_frac", type=float, default=0.1)
a = ap.parse_args()

CONFIG["snr_db"] = a.snr; CONFIG["channel_type"] = a.channel
CONFIG["D_jscc"] = a.D_jscc; CONFIG["jscc_spatial"] = True; CONFIG["jscc_grid"] = a.grid
CONFIG["refine_arch"] = "residual"; CONFIG["refine_disagreement_gate"] = False
DEV = "cuda:0"; os.makedirs(a.out, exist_ok=True)
print(f"[cfg] masked-diffusion denoiser  D_jscc={a.D_jscc} grid={a.grid}  {a.channel} {a.snr}dB  "
      f"tmax={a.tmax}", flush=True)

ZT=[]; ZH=[]; DIST=[]
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f)
    ZT.append(d["ZT"]); ZH.append(d["ZH"])
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT=torch.cat(ZT).float().reshape(-1,4,28,28); ZH=torch.cat(ZH).float().reshape(-1,4,28,28)
DIST=torch.cat(DIST).float(); N=len(ZT)
g=torch.Generator().manual_seed(0); perm=torch.randperm(N, generator=g)
nval=int(N*a.val_frac); va=perm[:nval]; tr=perm[nval:]
print(f"[data] {N} frames  disturbed={int(DIST.sum())} ({100*DIST.mean():.1f}%)  train {len(tr)} val {len(va)}", flush=True)

sysm = SemComSystem(CONFIG).to(DEV)
rd = sysm.refinement_diffusion
opt = torch.optim.AdamW(sysm.parameters(), lr=a.lr)
Tmax = min(a.tmax, len(rd.alphas_cumprod)-1)

def run_epoch(idx, train):
    sysm.train(train)
    order = idx[torch.randperm(len(idx))] if train else idx
    L=0.0; nb=0
    for i in range(0, len(order), a.bs):
        sel=order[i:i+a.bs]
        z=ZT[sel].to(DEV); h=ZH[sel].to(DEV); dm=DIST[sel].to(DEV); B=len(z)
        with torch.set_grad_enabled(train):
            _,_,s = sysm.jscc_encoder(z, sample=train); st = sysm.channel(s)
            # s~ dropout for classifier-free guidance: some samples see NO channel
            if a.sdrop > 0:
                keep = (torch.rand(B, device=DEV) >= a.sdrop).float().view(B,1)
                st = st * keep
            # HIGH-noise-weighted t: force a fraction to [Tmax/2, Tmax] where x_t is ~pure noise
            # and cannot leak z_t, so the model must decode the cube from s~.
            t_lo = torch.randint(1, Tmax+1, (B,), device=DEV)
            t_hi = torch.randint(Tmax//2, Tmax+1, (B,), device=DEV)
            hi = (torch.rand(B, device=DEV) < a.hi_noise_frac)
            t = torch.where(hi, t_hi, t_lo)
            ab = rd.alphas_cumprod[t].view(B,1,1,1).float()
            x_t = ab.sqrt()*z + (1.0-ab).sqrt()*torch.randn_like(z)      # noise the TARGET z_t (DDPM)
            pred = rd._denoise_x0(x_t, t, h, st)                          # predict x0 = z_t
            ds = ((pred - z)**2).mean(dim=(1,2,3))
            fw = 1.0 + a.dist_weight*dm
            loss = (fw*ds).sum()/fw.sum()
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(sysm.parameters(), 5.0); opt.step()
        L+=loss.item(); nb+=1
    return L/nb

for e in range(1, a.epochs+1):
    tl=run_epoch(tr, True); vl=run_epoch(va, False)
    print(f"ep{e:>2}/{a.epochs}  train {tl:.4f}  val {vl:.4f}", flush=True)
    torch.save({"system_state": sysm.state_dict(),
                "config": {k: CONFIG[k] for k in ("snr_db","channel_type","D_jscc","jscc_spatial","jscc_grid")}},
               os.path.join(a.out, "masked.pt"))
print("DONE", flush=True)
