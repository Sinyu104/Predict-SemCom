"""Why is the reconstruction blurry — is the cube MOVED, or DISSOLVED?

`recall 0.49 / shift 19.6 px` is ambiguous.  A binary colour mask reports the same
numbers for two completely different failures:

    MOVED     the cube is sharp and in the wrong place  -> encoder/rate problem
    DISSOLVED the cube is in the right place but faded below the mask threshold,
              so what survives is a fringe whose centroid wanders -> objective problem

Those have opposite fixes, so the mask alone cannot direct the work.  This script
separates them by measuring the cube's colour CONTRAST (how far the winning channel
beats the others, before any threshold) and the image's high-frequency energy,
alongside the mask stats.

It also runs the ladder that isolates which stage loses it, on identical frames:

    vae        z_t -> decode                    the codec floor; nothing beats this
    zhat       Ctrl-World prediction            what Stage 2 starts from
    full       refine(zhat, s~)                 the deployed path
    oracle_z   refine(z_t,  s~)                 PERFECT predictor handed to the refiner.
                                                If this is still blurry the refiner is
                                                destroying information, and no amount of
                                                Stage-1 work can help.
    oracle_ch  refine(zhat, s) at ~60 dB        noiseless channel. Separates channel
                                                noise from decoder behaviour.

    python experiments/stage2_twophase/blur_autopsy.py \
        --pre outputs/rollout_anch16_s2 --ckpt outputs/stage2_lam0_bs32/phase2.pt --n 768
"""
import os, sys, glob, pathlib, argparse, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                       # noqa: E402
from models import SemComSystem                 # noqa: E402
from vae_wrapper import VAEWrapper              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=768, help="(episode, t) pairs to score")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--tmin", type=int, default=16)
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--horizon_K", type=int, default=0,
                help="K the rollout was collected at. One Ctrl-World call emits K frames, "
                     "so cache index j is (j %% K)+1 steps past the newest history frame "
                     "and the aggregate mixes all K horizons. Set this to break z_hat out "
                     "per horizon, which separates 'cannot predict' from 'cannot "
                     "extrapolate 8 steps'. 0 = off.")
ap.add_argument("--phase_bins", type=int, default=0,
                help="Bucket z_hat by position WITHIN the episode. An aggregate over a "
                     "short rollout window is not the episode: measured over frames "
                     "41-88 z_hat scored recall 0.99, while the rendered frames at 150 "
                     "and 200 had every cube erased. 0 = off.")
a = ap.parse_args()

DEV = a.device
MIN_CUBE_PX, MAX_CUBE_PX = 25, 2000     # below: not visible;  above: it's the table
# Colour-dominance score per colour: how far the defining channels beat the rest.
# The binary masks in cube_geometry.py are thresholds ON THIS, so reporting the raw
# score shows whether a lost cube fell below the threshold or moved away from it.
DOM = {
    "orange":  lambda r, g, b: np.minimum(r - b, 2.0 * (r - g)),
    "magenta": lambda r, g, b: np.minimum(r - g, b - g),
    "yellow":  lambda r, g, b: np.minimum(r - b, g - b),
}


def split(im):
    return im[..., 0], im[..., 1], im[..., 2]


def cube_masks(im):
    r, g, b = split(im)
    return {
        "orange":  (r > 110) & (r - b > 50) & (r - g > 15) & (g - b > 10),
        "magenta": (r > 90) & (b > 90) & (r - g > 50) & (b - g > 40),
        "yellow":  (r > 130) & (g > 120) & (r - b > 60) & (np.abs(r - g) < 45),
    }


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([ys.mean(), xs.mean()])


def hf_energy(im):
    """Mean |Laplacian| of luma over the frame: a blur meter that needs no reference."""
    y = im.mean(-1)
    lap = (4 * y[:, 1:-1, 1:-1] - y[:, :-2, 1:-1] - y[:, 2:, 1:-1]
           - y[:, 1:-1, :-2] - y[:, 1:-1, 2:])
    return np.abs(lap).mean(axis=(1, 2))


ZT, ZH = [], []
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f, map_location="cpu"); ZT.append(d["ZT"]); ZH.append(d["ZH"])
ZT = torch.cat(ZT); ZH = torch.cat(ZH)          # (E, T, 4, 28, 28)
E, T = ZT.shape[:2]
pairs = [(e, t) for e in range(E) for t in range(a.tmin, T)]
g = torch.Generator().manual_seed(a.seed)
pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=g)[:a.n].tolist()]
print(f"[data] {E} episodes x {T} steps; scoring {len(pairs)} frames (seed {a.seed})")

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
dec = lambda z: vae.decode(z.to(DEV).float()).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.

sysm = SemComSystem(CONFIG).to(DEV)
sk = torch.load(a.ckpt, map_location="cpu")
res = sysm.load_state_dict(sk.get("system_state", sk), strict=False)
trained = {k for k in sk.get("system_state", sk)}
missing = [k for k in res.missing_keys if not k.endswith(("pos_embed", "sig_pos_buf"))]
print(f"[ckpt] {len(trained)} tensors in file; missing={missing or 'none'}; "
      f"unexpected={res.unexpected_keys or 'none'}")
if missing:
    print("[ckpt] !! parameters above were left at random init — every number below is void")
sysm.eval()

# Noiseless channel for the oracle_ch arm, built once.
from models import FadingChannel                # noqa: E402
clean_ch = FadingChannel(60.0, channel_type=CONFIG.get("channel_type", "rayleigh")).to(DEV)

ARMS = ["vae", "zhat", "full", "oracle_z", "oracle_ch", "sig_only"]
# The refiner returns `z_hat + delta`, so an arm fed z_hat = z_t scores well even if
# the network emits nothing at all -- it is the input, not a reconstruction.  `sig_only`
# zeroes the prior, making the output exactly `delta`: everything the channel carried
# and nothing else.  That is the only arm that measures Stage 2's reconstruction power
# rather than its pass-through.  `dnorm` below reports ||delta|| per arm for the same
# reason: it says how much work the network actually did.
dnorm = {v: [] for v in ARMS}
# per-horizon z_hat stats, filled only when --horizon_K > 0
hstat = {}
pstat = {}          # per episode-phase z_hat stats, --phase_bins > 0
stats = {v: {"rec": [], "sh": [], "miss": 0, "n": 0, "dom": [], "hf": []} for v in ARMS}
truth_dom, truth_hf = [], []
# `shift` is only defined where the arm's cube SURVIVED, so each arm's mean is taken
# over a different set of frames -- an arm that drops every hard frame scores a
# flattering shift on the easy ones it kept.  Record per-frame shifts so the arms can
# also be compared on the frames where BOTH survived, which is the honest comparison.
per_frame = {v: [] for v in ARMS}          # None where the arm lost the cube

with torch.no_grad():
    for i in range(0, len(pairs), a.bs):
        chunk = pairs[i:i + a.bs]
        ep = torch.tensor([e for e, _ in chunk]); ts = torch.tensor([t for _, t in chunk])
        z = ZT[ep, ts].to(DEV).float(); h = ZH[ep, ts].to(DEV).float(); B = len(chunk)
        t0 = torch.zeros(B, dtype=torch.long, device=DEV)
        rd = sysm.refinement_diffusion

        _, _, s = sysm.jscc_encoder(z, sample=False)
        st = sysm.channel(s)
        zero = torch.zeros_like(h)
        out = {
            "vae":       z,
            "zhat":      h,
            "full":      rd._denoise_x0(h, t0, h, st),
            "oracle_z":  rd._denoise_x0(z, t0, z, st),
            "oracle_ch": rd._denoise_x0(h, t0, h, clean_ch(s)),
            "sig_only":  rd._denoise_x0(zero, t0, zero, st),
        }
        base = {"vae": z, "zhat": h, "full": h, "oracle_z": z, "oracle_ch": h,
                "sig_only": zero}
        for v in ARMS:
            dnorm[v].append((out[v] - base[v]).pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy())
        im_t = dec(z)
        mt = cube_masks(im_t)
        r, gg, b = split(im_t)
        dom_t = {k: DOM[k](r, gg, b) for k in DOM}
        truth_hf.append(hf_energy(im_t))

        ims = {v: (im_t if v == "vae" else dec(out[v])) for v in ARMS}
        for v in ARMS:
            stats[v]["hf"].append(hf_energy(ims[v]))
        mrs = {v: cube_masks(ims[v]) for v in ARMS}
        doms = {}
        for v in ARMS:
            r2, g2, b2 = split(ims[v])
            doms[v] = {k: DOM[k](r2, g2, b2) for k in DOM}

        for k in mt:
            for bi in range(B):
                A = mt[k][bi]
                na = int(A.sum())
                if not (MIN_CUBE_PX <= na <= MAX_CUBE_PX):
                    continue
                # Contrast is read INSIDE the true cube footprint for every arm, so a
                # faded cube is measured where it should be rather than wherever its
                # surviving fringe happens to land.
                truth_dom.append(float(dom_t[k][bi][A].mean()))
                if a.horizon_K > 0:
                    hz = int(ts[bi]) % a.horizon_K + 1
                    hd = hstat.setdefault(hz, {"rec": [], "dom": [], "miss": 0, "n": 0,
                                               "sh": []})
                    Bh = mrs["zhat"][k][bi]; nbh = int(Bh.sum())
                    hd["n"] += 1
                    hd["rec"].append(nbh / na)
                    hd["dom"].append(float(doms["zhat"][k][bi][A].mean()))
                    if nbh < 10:
                        hd["miss"] += 1
                    else:
                        hd["sh"].append(float(np.hypot(*(centroid(A) - centroid(Bh)))))
                if a.phase_bins > 0:
                    pb = min(int(ts[bi]) * a.phase_bins // T, a.phase_bins - 1)
                    pd = pstat.setdefault(pb, {"rec": [], "dom": [], "miss": 0, "n": 0,
                                               "sh": [], "lo": T, "hi": 0})
                    Bp = mrs["zhat"][k][bi]; nbp = int(Bp.sum())
                    pd["n"] += 1
                    pd["lo"] = min(pd["lo"], int(ts[bi])); pd["hi"] = max(pd["hi"], int(ts[bi]))
                    pd["rec"].append(nbp / na)
                    pd["dom"].append(float(doms["zhat"][k][bi][A].mean()))
                    if nbp < 10:
                        pd["miss"] += 1
                    else:
                        pd["sh"].append(float(np.hypot(*(centroid(A) - centroid(Bp)))))
                for v in ARMS:
                    d = stats[v]
                    d["n"] += 1
                    d["dom"].append(float(doms[v][k][bi][A].mean()))
                    Bm = mrs[v][k][bi]
                    nb = int(Bm.sum())
                    d["rec"].append(nb / na)
                    if nb < 10:
                        d["miss"] += 1
                        per_frame[v].append(None)
                    else:
                        sh_ = float(np.hypot(*(centroid(A) - centroid(Bm))))
                        d["sh"].append(sh_)
                        per_frame[v].append(sh_)
        if i % (a.bs * 10) == 0:
            print(f"  {i}/{len(pairs)}", flush=True)

tdom = float(np.mean(truth_dom)); thf = float(np.concatenate(truth_hf).mean())
print(f"\ntruth: cube colour contrast {tdom:.1f}   frame HF energy {thf:.2f}")
print("(contrast is measured inside the TRUE cube footprint; the mask threshold is ~45-50,")
print(" so an arm whose contrast falls under that has DISSOLVED the cube, not moved it)\n")
zrms = float(np.sqrt(np.mean([ZT[e, t].float().pow(2).mean().item() for e, t in pairs[:256]])))
print(f"{'arm':<10} {'n':>5} {'recall':>7} {'shift px':>9} {'missing':>8} "
      f"{'contrast':>9} {'vs truth':>9} {'HF':>7} {'vs truth':>9} {'||delta||':>10}")
for v in ARMS:
    d = stats[v]
    sh = np.mean(d["sh"]) if d["sh"] else float("nan")
    dm = float(np.mean(d["dom"])); hf = float(np.concatenate(d["hf"]).mean())
    dn = float(np.concatenate(dnorm[v]).mean())
    print(f"{v:<10} {d['n']:5d} {np.mean(d['rec']):7.2f} {sh:9.2f} "
          f"{100 * d['miss'] / max(d['n'], 1):7.1f}% {dm:9.1f} {100*dm/tdom:8.0f}% "
          f"{hf:7.2f} {100*hf/thf:8.0f}% {dn:10.3f}")
print(f"\n||delta|| is the RMS of what the network ADDED to its base; latent RMS = {zrms:.3f}.")
print("If oracle_z's delta is near zero, that arm is the input passed through and proves")
print("only that the refiner does no harm.  sig_only's score IS delta: the channel alone.")

if pstat:
    print(f"\nz_hat BY EPISODE PHASE (rollout step within the episode):")
    print(f"{'frames':<12} {'n':>5} {'recall':>7} {'shift px':>9} {'missing':>8} "
          f"{'contrast':>9} {'vs truth':>9}")
    for pb in sorted(pstat):
        d = pstat[pb]
        sh = np.mean(d["sh"]) if d["sh"] else float("nan")
        dm = float(np.mean(d["dom"]))
        span = f"{d['lo']}-{d['hi']}"
        print(f"{span:<12} {d['n']:5d} {np.mean(d['rec']):7.2f} {sh:9.2f} "
              f"{100 * d['miss'] / max(d['n'], 1):7.1f}% {dm:9.1f} {100*dm/tdom:8.0f}%")

if hstat:
    print(f"\nz_hat BY HORIZON (steps past the newest history frame, K={a.horizon_K}):")
    print(f"{'h':<4} {'n':>5} {'recall':>7} {'shift px':>9} {'missing':>8} "
          f"{'contrast':>9} {'vs truth':>9}")
    for hz in sorted(hstat):
        d = hstat[hz]
        sh = np.mean(d["sh"]) if d["sh"] else float("nan")
        dm = float(np.mean(d["dom"]))
        print(f"{hz:<4} {d['n']:5d} {np.mean(d['rec']):7.2f} {sh:9.2f} "
              f"{100 * d['miss'] / max(d['n'], 1):7.1f}% {dm:9.1f} {100*dm/tdom:8.0f}%")

both = [i for i in range(len(per_frame["zhat"]))
        if per_frame["zhat"][i] is not None and per_frame["full"][i] is not None]
print(f"\nMATCHED on the {len(both)} frames where zhat AND full both kept the cube "
      f"(of {len(per_frame['zhat'])}):")
for v in ("zhat", "full"):
    print(f"  {v:<6} shift {np.mean([per_frame[v][i] for i in both]):6.2f} px")
print("  Compare with the unmatched column above: zhat's shift is averaged over only the")
print("  frames it did not drop, i.e. the easy ones, so the two are not the same number.")

print("\nvae is the floor: no arm can beat it. If `full` contrast is far below `zhat`,")
print("the refiner is the thing erasing the cube. If oracle_z is also low, it erases it")
print("even when handed the exact answer, and the objective -- not the predictor -- is why.")
print("DONE")
