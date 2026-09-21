"""Geometry metric for Stage-2 checkpoints: does the cube end up in the RIGHT PLACE?

Latent MSE cannot see the failure that stops the robot grasping.  Measured live
with a perfect predictor context (clean history, no drift), the target cube was
displaced 12.9 px -- about its own width -- and vanished entirely on some frames,
while latent MSE sat at a value that looked fine.  A policy aiming where it sees
the cube misses by one cube.

Reports, per colour:
    recall   = recon cube px / true cube px     1.0 = right size, 0 = erased
    shift    = centroid displacement in px      cube is ~15 px wide
    missing  = fraction of frames where the cube is in the truth but not the recon

BLUE IS EXCLUDED ON PURPOSE.  The table is blue and covers ~35,000 px, so a blue
colour-dominance mask measures the table, not the cube: an earlier version of this
metric reported a reassuring 1.45 px "shift" that was the table's centroid and
meant nothing.  Same class of error as the saturation mask that caught the table
earlier in this work.  The guard below drops any mask above MAX_CUBE_PX.

    python experiments/stage2_twophase/cube_geometry.py \
        --pre outputs/rollout_anch16_s2 --n 512 \
        --ckpt lam0=outputs/stage2_lam0_bs32/phase2.pt
"""
import os, sys, glob, pathlib, argparse, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                       # noqa: E402
from models import SemComSystem                 # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--ckpt", nargs="+", required=True, help="label=path ...")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--spatial", action="store_true",
                help="Checkpoint uses the spatial JSCC code. Must match how it was trained; "
                     "the state dict will not load otherwise.")
ap.add_argument("--grid", type=int, default=8)
ap.add_argument("--t2max", type=int, default=0)
ap.add_argument("--eval_steps", type=int, default=10)
a = ap.parse_args()
CONFIG["jscc_spatial"] = a.spatial
CONFIG["jscc_grid"]    = a.grid

DEV = "cuda:0"
MIN_CUBE_PX, MAX_CUBE_PX = 25, 2000     # below: not visible;  above: it's the table

def cube_masks(im):
    """im (B,H,W,3) float 0..255 -> {colour: (B,H,W) bool}.  Blue omitted, see docstring."""
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {
        "orange":  (r > 110) & (r - b > 50) & (r - g > 15) & (g - b > 10),
        "magenta": (r > 90) & (b > 90) & (r - g > 50) & (b - g > 40),
        "yellow":  (r > 130) & (g > 120) & (r - b > 60) & (np.abs(r - g) < 45),
    }

def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([ys.mean(), xs.mean()])

ZT, ZH = [], []
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f); ZT.append(d["ZT"]); ZH.append(d["ZH"])
ZT = torch.cat(ZT).float().reshape(-1, 4, 28, 28)
ZH = torch.cat(ZH).float().reshape(-1, 4, 28, 28)
g = torch.Generator().manual_seed(a.seed)
sub = torch.randperm(len(ZT), generator=g)[:a.n]
print(f"[data] {len(ZT)} samples; common random draw of {len(sub)} (seed {a.seed})")

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
dec = lambda t: vae.decode(t).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.

for spec in a.ckpt:
    label, path = spec.split("=", 1)
    sysm = SemComSystem(CONFIG).to(DEV)
    sk = torch.load(path, map_location="cpu")
    sysm.load_state_dict(sk.get("system_state", sk), strict=False)
    sysm.eval()

    mse_all, stats = [], {}
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
            mse_all.append(((p - z) ** 2).mean(dim=(1, 2, 3)).cpu())
            mt, mr = cube_masks(dec(z)), cube_masks(dec(p))
            for k in mt:
                for bi in range(B):
                    A, Bm = mt[k][bi], mr[k][bi]
                    na = int(A.sum())
                    if not (MIN_CUBE_PX <= na <= MAX_CUBE_PX):
                        continue                      # not visible, or it's the table
                    d = stats.setdefault(k, {"rec": [], "sh": [], "miss": 0, "n": 0})
                    d["n"] += 1
                    nb = int(Bm.sum())
                    d["rec"].append(nb / na)
                    if nb < 10:
                        d["miss"] += 1
                    else:
                        d["sh"].append(float(np.hypot(*(centroid(A) - centroid(Bm)))))

    mse_all = torch.cat(mse_all).numpy()
    print(f"\n=== {label}   latent MSE = {mse_all.mean():.5f} ===")
    print(f"{'colour':<9} {'n':>5} {'recall':>7} {'shift px':>9} {'missing':>8}")
    for k, d in stats.items():
        sh = np.mean(d["sh"]) if d["sh"] else float("nan")
        print(f"{k:<9} {d['n']:5d} {np.mean(d['rec']):7.2f} {sh:9.2f} "
              f"{100*d['miss']/max(d['n'],1):7.1f}%")
    del sysm; torch.cuda.empty_cache()

print("\nrecall 1.0 = right size, 0 = erased.  shift: cube is ~15 px wide, so a shift")
print("near 15 means the policy aims one full cube away.  blue excluded (it is the table).")
print("DONE")
