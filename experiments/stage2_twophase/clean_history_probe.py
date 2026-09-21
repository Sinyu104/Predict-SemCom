"""Is Stage 1 broken, or is it being RUN in a configuration that breaks it?

The z_hat scored by blur_autopsy.py came from a closed-loop rollout: the history
fed to Ctrl-World was the receiver's own reconstruction, drifting, at
history_stride=8, and every prediction was 1..8 steps past the newest history
frame.  Three separate things, measured as one number.

This runs Ctrl-World on CLEAN ground-truth history and scores each horizon step
on its own, so drift and horizon separate:

    h=1 ... h=8     prediction i steps past the newest history frame
    copy            reuse that newest history frame, unchanged -- free, no model

If h=1 on clean history has a sharp cube, Stage 1 is not broken; it is being asked
to extrapolate 8 steps from a drifted history.  If h=1 is already gone, it is the
model.  `copy` is the floor any predictor has to clear to be worth running at all.

    python experiments/stage2_twophase/clean_history_probe.py --eps_per_task 8 --anchors 4
"""
import os, sys, glob, pathlib, argparse, numpy as np, torch, h5py

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                       # noqa: E402
from vae_wrapper import VAEWrapper              # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/stage1_stride_k8/stage1_rollout_ep1.pt")
ap.add_argument("--stride", type=int, default=8, help="history_stride the ckpt was trained at")
ap.add_argument("--K", type=int, default=8, help="num_pred the ckpt was trained at")
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--eps_per_task", type=int, default=8)
ap.add_argument("--anchors", type=int, default=4, help="rollout starts sampled per episode")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = a.device
T_H = CONFIG["num_history"]; M = a.stride; K = a.K
SPAN = (T_H - 1) * M + 1                 # clean frames needed before an anchor
MIN_CUBE_PX, MAX_CUBE_PX = 25, 2000
DOM = {
    "orange":  lambda r, g, b: np.minimum(r - b, 2.0 * (r - g)),
    "magenta": lambda r, g, b: np.minimum(r - g, b - g),
    "yellow":  lambda r, g, b: np.minimum(r - b, g - b),
}


def cube_masks(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {
        "orange":  (r > 110) & (r - b > 50) & (r - g > 15) & (g - b > 10),
        "magenta": (r > 90) & (b > 90) & (r - g > 50) & (b - g > 40),
        "yellow":  (r > 130) & (g > 120) & (r - b > 60) & (np.abs(r - g) < 45),
    }


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([ys.mean(), xs.mean()])


vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck
cw.eval()
print(f"[cfg] {a.ckpt}  T_H={T_H} stride={M} K={K} span={SPAN} ddim={a.ddim}")

dec = lambda z: vae.decode(z.to(DEV).float()).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.

ARMS = [f"h={i+1}" for i in range(K)] + ["copy"]
stats = {v: {"rec": [], "sh": [], "miss": 0, "n": 0, "dom": []} for v in ARMS}
truth_dom = []
rng = np.random.default_rng(a.seed)
ncalls = 0

for task in a.tasks:
    p = os.path.join(REPO, "data", task, "demos_disturbed.hdf5")
    if not os.path.isfile(p):
        print(f"  skip {task}: missing"); continue
    with h5py.File(p, "r") as f:
        eps = sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit())))
        for e in eps[:a.eps_per_task]:
            g = f[e]
            n = len(g["observations_cam1"])
            if n < SPAN + K + 2:
                continue
            lo, hi = SPAN, n - K - 1
            if hi <= lo:
                continue
            for t in rng.choice(np.arange(lo, hi), size=min(a.anchors, hi - lo),
                                replace=False):
                t = int(t)
                # CLEAN history: the true frames at t-1, t-1-M, ... oldest first.
                hidx = [t - 1 - i * M for i in range(T_H)][::-1]
                need = sorted(set(hidx + list(range(t, t + K))))
                x = torch.from_numpy(
                    np.array(g["observations_cam1"][need])).float().div_(255.).permute(0, 3, 1, 2)
                pos = {fr: i for i, fr in enumerate(need)}
                act = torch.from_numpy(np.array(g["actions"][:t + K], dtype=np.float32))
                with torch.no_grad():
                    Z = torch.cat([vae.encode(x[j:j + 4].to(DEV)) for j in range(0, len(x), 4)])
                    H = torch.stack([Z[pos[fr]] for fr in hidx], 0).unsqueeze(0)
                    a_win = torch.cat([act[torch.tensor(hidx)], act[t:t + K]], 0
                                      ).unsqueeze(0).to(DEV)
                    zp = cw.predict_next_latent(H, a_win, n_steps=a.ddim).float()  # (1,K,C,H,W)
                ncalls += 1
                last_hist = Z[pos[hidx[-1]]].unsqueeze(0)          # the `copy` baseline
                for i in range(K):
                    truth = Z[pos[t + i]].unsqueeze(0)
                    im_t = dec(truth)
                    mt = cube_masks(im_t)
                    r, gg, b = im_t[..., 0], im_t[..., 1], im_t[..., 2]
                    dom_t = {k: DOM[k](r, gg, b) for k in DOM}
                    cands = {f"h={i+1}": zp[:, i], "copy": last_hist}
                    ims = {v: dec(c) for v, c in cands.items()}
                    for k in mt:
                        A = mt[k][0]
                        na = int(A.sum())
                        if not (MIN_CUBE_PX <= na <= MAX_CUBE_PX):
                            continue
                        truth_dom.append(float(dom_t[k][0][A].mean()))
                        for v, im in ims.items():
                            r2, g2, b2 = im[..., 0], im[..., 1], im[..., 2]
                            d = stats[v]
                            d["n"] += 1
                            d["dom"].append(float(DOM[k](r2, g2, b2)[0][A].mean()))
                            Bm = cube_masks(im)[k][0]
                            nb = int(Bm.sum())
                            d["rec"].append(nb / na)
                            if nb < 10:
                                d["miss"] += 1
                            else:
                                d["sh"].append(float(np.hypot(*(centroid(A) - centroid(Bm)))))
    print(f"  {task}: {ncalls} prediction calls so far", flush=True)

tdom = float(np.mean(truth_dom))
print(f"\n{ncalls} Ctrl-World calls on CLEAN history.  truth cube contrast {tdom:.1f}")
print(f"{'arm':<8} {'n':>5} {'recall':>7} {'shift px':>9} {'missing':>8} "
      f"{'contrast':>9} {'vs truth':>9}")
for v in ARMS:
    d = stats[v]
    if not d["n"]:
        continue
    sh = np.mean(d["sh"]) if d["sh"] else float("nan")
    dm = float(np.mean(d["dom"]))
    print(f"{v:<8} {d['n']:5d} {np.mean(d['rec']):7.2f} {sh:9.2f} "
          f"{100 * d['miss'] / max(d['n'], 1):7.1f}% {dm:9.1f} {100*dm/tdom:8.0f}%")

print("\nCompare h=8 against the closed-loop number (recall 0.26, contrast -19.3): the gap")
print("is drift.  Compare h=1 against h=8: that gap is the horizon.  Compare both against")
print("`copy`: a predictor below it is worse than not predicting.")
print("DONE")
