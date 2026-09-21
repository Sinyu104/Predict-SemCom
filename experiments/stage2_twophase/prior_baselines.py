"""Is z_hat actually a better prior than doing nothing?

Every Stage-2 result in this project is measured against z_hat as the starting
point, and z_hat is assumed to be the best prior available to the receiver.  That
assumption was never tested against the cheapest possible alternative: the
receiver already holds a reconstruction of frame t-1, so "copy the previous
frame" is free side information.

The rollout cache is SEQUENTIAL (episodes x 150 steps), so the baselines come out
of it with no model and no recompute:

    zhat      Ctrl-World's prediction               (the current design)
    prev1     z_{t-1}                               free; what the receiver holds
    prev8     z_{t-8}                               the newest frame Ctrl-World is
                                                    actually shown (history_stride=8)

Scored with the cube_geometry metric, not MSE, because MSE is blind to the
failure that stops the grasp.  Frames are split into MOVING and STILL by how far
the true cube travelled between t-1 and t, since a static-scene prior looks
perfect on still frames and tells you nothing.

    python experiments/stage2_twophase/prior_baselines.py \
        --pre outputs/rollout_anch16_s2 --n 3000
"""
import os, sys, glob, pathlib, argparse, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                       # noqa: E402
from vae_wrapper import VAEWrapper              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--n", type=int, default=3000, help="(episode, t) pairs to score")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--tmin", type=int, default=16, help="skip the opening frames")
ap.add_argument("--move_px", type=float, default=1.0,
                help="true-cube travel between t-1 and t above which a frame is MOVING")
ap.add_argument("--device", default="cuda:3")
a = ap.parse_args()

DEV = a.device
MIN_CUBE_PX, MAX_CUBE_PX = 25, 2000     # below: not visible;  above: it's the table


def cube_masks(im):
    """im (B,H,W,3) float 0..255 -> {colour: (B,H,W) bool}.  Blue omitted: it is the table."""
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
    d = torch.load(f, map_location="cpu"); ZT.append(d["ZT"]); ZH.append(d["ZH"])
ZT = torch.cat(ZT); ZH = torch.cat(ZH)          # (E, T, 4, 28, 28)
E, T = ZT.shape[:2]
print(f"[data] {E} episodes x {T} steps")

pairs = [(e, t) for e in range(E) for t in range(a.tmin, T)]
g = torch.Generator().manual_seed(a.seed)
sel = torch.randperm(len(pairs), generator=g)[:a.n]
pairs = [pairs[i] for i in sel.tolist()]
print(f"[data] scoring {len(pairs)} (episode, t) pairs (seed {a.seed})")

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
dec = lambda t_: vae.decode(t_.to(DEV).float()).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.

VARIANTS = ["zhat", "prev1", "prev8"]
# stats[variant][regime][colour] -> lists
stats = {v: {r: {} for r in ("MOVING", "STILL")} for v in VARIANTS}
seen = {r: 0 for r in ("MOVING", "STILL")}

with torch.no_grad():
    for i in range(0, len(pairs), a.bs):
        chunk = pairs[i:i + a.bs]
        ep = torch.tensor([e for e, _ in chunk]); ts = torch.tensor([t for _, t in chunk])
        truth = ZT[ep, ts]
        cand = {
            "zhat":  ZH[ep, ts],
            "prev1": ZT[ep, (ts - 1).clamp(min=0)],
            "prev8": ZT[ep, (ts - 8).clamp(min=0)],
        }
        m_true = cube_masks(dec(truth))
        m_prev = cube_masks(dec(ZT[ep, (ts - 1).clamp(min=0)]))     # for the motion split
        m_cand = {v: cube_masks(dec(c)) for v, c in cand.items()}

        for k in m_true:
            for bi in range(len(chunk)):
                A = m_true[k][bi]
                na = int(A.sum())
                if not (MIN_CUBE_PX <= na <= MAX_CUBE_PX):
                    continue
                P = m_prev[k][bi]
                if not (MIN_CUBE_PX <= int(P.sum()) <= MAX_CUBE_PX):
                    continue                       # cannot classify the regime
                travel = float(np.hypot(*(centroid(A) - centroid(P))))
                reg = "MOVING" if travel > a.move_px else "STILL"
                seen[reg] += 1
                for v in VARIANTS:
                    Bm = m_cand[v][k][bi]
                    d = stats[v][reg].setdefault(
                        k, {"rec": [], "sh": [], "miss": 0, "n": 0})
                    d["n"] += 1
                    nb = int(Bm.sum())
                    d["rec"].append(nb / na)
                    if nb < 10:
                        d["miss"] += 1
                    else:
                        d["sh"].append(float(np.hypot(*(centroid(A) - centroid(Bm)))))
        if i % (a.bs * 20) == 0:
            print(f"  {i}/{len(pairs)}", flush=True)

for reg in ("MOVING", "STILL"):
    print(f"\n=== {reg}  ({seen[reg]} colour-frames) ===")
    print(f"{'prior':<7} {'colour':<9} {'n':>5} {'recall':>7} {'shift px':>9} {'missing':>8}")
    for v in VARIANTS:
        agg_sh, agg_rec, agg_miss, agg_n = [], [], 0, 0
        for k, d in sorted(stats[v][reg].items()):
            sh = np.mean(d["sh"]) if d["sh"] else float("nan")
            print(f"{v:<7} {k:<9} {d['n']:5d} {np.mean(d['rec']):7.2f} {sh:9.2f} "
                  f"{100 * d['miss'] / max(d['n'], 1):7.1f}%")
            agg_sh += d["sh"]; agg_rec += d["rec"]; agg_miss += d["miss"]; agg_n += d["n"]
        if agg_n:
            print(f"{v:<7} {'ALL':<9} {agg_n:5d} {np.mean(agg_rec):7.2f} "
                  f"{(np.mean(agg_sh) if agg_sh else float('nan')):9.2f} "
                  f"{100 * agg_miss / agg_n:7.1f}%   <--")
        print()

print("cube is ~15 px wide.  A prior that beats zhat here is free side information "
      "that the current design throws away.")
print("DONE")
