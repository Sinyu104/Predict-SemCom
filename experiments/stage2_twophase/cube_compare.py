"""Score finished Stage-2 checkpoints on ONE common random subsample.

twophase.py reports cube IoU inside each run, but two runs only compare if they
scored the same frames.  This loads several phase2.pt checkpoints and evaluates
them all against an identical seeded random draw, reporting unweighted MSE (so
--lam runs stay comparable to lam=0) alongside cube overlap.

    python experiments/stage2_twophase/cube_compare.py \
        --pre outputs/rollout_anch16_s2 --n 512 \
        --ckpt lam0=outputs/stage2_lam0_bs32/phase2.pt \
               lam3=outputs/stage2_lam3_bs32/phase2.pt \
               lam8=outputs/stage2_lam8_bs32/phase2.pt
"""
import os, sys, glob, pathlib, argparse, numpy as np, torch
REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG
from models import SemComSystem
from vae_wrapper import VAEWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--ckpt", nargs="+", required=True, help="label=path ...")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--cube_color", default="red", choices=["red", "green", "blue"])
ap.add_argument("--t2max", type=int, default=0)
ap.add_argument("--eval_steps", type=int, default=10)
a = ap.parse_args()

DEV = "cuda:0"
ZT, ZH = [], []
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f); ZT.append(d["ZT"]); ZH.append(d["ZH"])
ZT = torch.cat(ZT).float().reshape(-1, 4, 28, 28)
ZH = torch.cat(ZH).float().reshape(-1, 4, 28, 28)

g = torch.Generator().manual_seed(a.seed)
sub = torch.randperm(len(ZT), generator=g)[:a.n]
dev_all = ((ZT[sub] - ZH[sub]) ** 2).mean(dim=(1, 2, 3))
print(f"[data] {len(ZT)} samples, scoring a common random draw of {len(sub)} "
      f"(seed {a.seed})   mean dev ||z-zhat||^2 = {dev_all.mean():.4f}")

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
C_IDX = {"red": 0, "green": 1, "blue": 2}[a.cube_color]
OTH = [i for i in range(3) if i != C_IDX]

def mask(im):
    return (im[..., C_IDX] > 90) & (im[..., C_IDX] - im[..., OTH[0]] > 45) \
                                 & (im[..., C_IDX] - im[..., OTH[1]] > 45)

print(f"\n{'run':<10} {'MSE(A)':>9} {'MSE(zhat)':>10} {'A vs C':>8} "
      f"{'cubeIoU':>8} {'px true':>8} {'px rec':>7} {'recall':>7}")
for spec in a.ckpt:
    label, path = spec.split("=", 1)
    sysm = SemComSystem(CONFIG).to(DEV)
    sk = torch.load(path, map_location="cpu")
    sysm.load_state_dict(sk.get("system_state", sk), strict=False)
    sysm.eval()
    A, U, G, R = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(sub), a.bs):
            sel = sub[i:i + a.bs]
            z = ZT[sel].to(DEV); h = ZH[sel].to(DEV); B = len(z)
            _, _, s = sysm.jscc_encoder(z, sample=False); st = sysm.channel(s)
            if a.t2max > 0:
                p = sysm.refinement_diffusion.sdedit_refine(
                    h, st, noise_level=a.t2max, n_steps=a.eval_steps)
            else:
                t0 = torch.zeros(B, dtype=torch.long, device=DEV)
                p = sysm.refinement_diffusion._denoise_x0(h, t0, h, st)
            A.append(((p - z) ** 2).mean(dim=(1, 2, 3)).cpu())
            dec = lambda t: vae.decode(t).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.
            mg, mr = mask(dec(z)), mask(dec(p))
            keep = mg.sum((1, 2)) > 20
            if keep.any():
                inter = (mg & mr).sum((1, 2))[keep]
                U.append(inter / np.maximum((mg | mr).sum((1, 2))[keep], 1))
                G.append(mg.sum((1, 2))[keep]); R.append(mr.sum((1, 2))[keep])
    A = torch.cat(A).numpy(); U = np.concatenate(U); G = np.concatenate(G); R = np.concatenate(R)
    c = dev_all.numpy()
    print(f"{label:<10} {A.mean():9.5f} {c.mean():10.5f} "
          f"{100*(c.mean()-A.mean())/c.mean():+7.1f}% {U.mean():8.3f} "
          f"{G.mean():8.0f} {R.mean():7.0f} {R.mean()/max(G.mean(),1):7.2f}")
    del sysm; torch.cuda.empty_cache()
print("\ncubeIoU = overlap of the cube mask;  recall = recon cube px / true cube px "
      "(1.0 = right size, 0 = erased)")
print("DONE")
