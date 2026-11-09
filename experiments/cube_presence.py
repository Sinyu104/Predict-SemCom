"""Is the TARGET cube present in the predicted +30 image?  Colour-count test on decoded base / wrist images.

For each window the target colour is read from the task string (blue / red / yellow tasks only).  The reference is the VAE
round trip of the true +30 latent (so decode artefacts cancel).  A prediction 'has the cube' if its target-colour pixel count
is >= 50% of the reference count (reference >= 30 px).  MOVING windows = the cube centroid shifts > 6 px between the last
history frame (copy) and the target, i.e. it is being carried / is near the gripper.

    python experiments/cube_presence.py --arms outputs/cube_check/arms300_orig2cam.npz outputs/cube_check/arms300_new_ep5.npz
"""
import argparse, numpy as np, matplotlib.colors as mc

ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", required=True); a = ap.parse_args()


def mask(img, col):
    h = mc.rgb_to_hsv(img.astype(np.float32) / 255); H, S, V = h[..., 0] * 360, h[..., 1], h[..., 2]
    if col == "blue":   return (H > 205) & (H < 250) & (S > 0.65) & (V > 0.35)
    if col == "red":    return ((H < 12) | (H > 348)) & (S > 0.6) & (V > 0.35)
    return (H > 40) & (H < 65) & (S > 0.55) & (V > 0.5)


def cen(m):
    ys, xs = np.nonzero(m); return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])


rows = {}
for pth in a.arms:
    d = np.load(pth, allow_pickle=True)
    names = [k[4:] for k in d.files if k.startswith("img_") and k[4:] not in ("vae", "copy")]
    tasks = [str(t) for t in d["task"]]
    for nm in ["copy"] + names:
        rows.setdefault(nm, [])
        for i, t in enumerate(tasks):
            col = next((c for c in ("blue", "red", "yellow") if c in t), None)
            if col is None: continue
            for view, pre in (("base", "img_"), ("wrist", "wimg_")):
                ref = mask(d[pre + "vae"][i], col); prd = mask(d[pre + nm][i], col); last = mask(d[pre + "copy"][i], col)
                nref, npd = ref.sum(), prd.sum()
                if nref < 30: continue
                mv = np.nanmax(np.abs(cen(ref) - cen(last))) > 6 if last.sum() >= 5 else False
                dist = np.linalg.norm(cen(prd) - cen(ref)) if npd >= 5 else np.nan
                rows[nm].append((col, view, int(mv), npd / nref, float(npd >= 0.5 * nref), dist))
print(f"{'model':14s} {'view':6s} {'windows':>8s} {'has cube':>9s} {'px ratio':>9s} {'centroid err px':>16s} | MOVING: n  has cube  px ratio")
for nm, r in rows.items():
    for view in ("base", "wrist"):
        x = [q for q in r if q[1] == view]
        if not x: continue
        mvx = [q for q in x if q[2]]
        print(f"{nm:14s} {view:6s} {len(x):8d} {np.mean([q[4] for q in x]):9.2f} {np.median([q[3] for q in x]):9.2f} "
              f"{np.nanmedian([q[5] for q in x]):16.1f} | {len(mvx):6d} {np.mean([q[4] for q in mvx]) if mvx else float('nan'):9.2f} "
              f"{np.median([q[3] for q in mvx]) if mvx else float('nan'):9.2f}")
    for col in ("blue", "red", "yellow"):
        x = [q for q in r if q[0] == col]
        if x: print(f"{'':14s} {col:6s} {len(x):8d} has cube {np.mean([q[4] for q in x]):.2f}")
