"""Does the predictor fail for lack of LOCAL VELOCITY, or because it cannot copy?

Measured at matched horizon, Ctrl-World is worse than persistence at h=1
(0.0600 vs 0.0317): one step ahead it corrupts a scene it could have copied.
Two causes are consistent with that, and they need different fixes:

  (a) no velocity signal.  History is spaced `stride` apart, so at stride=8 the
      two newest frames the model sees are t-1 and t-9.  A gripper crossing the
      frame in ~20 steps is aliased beyond recovery, so the model hedges and the
      small fast objects dissolve.  Fix = respace the history.
  (b) cannot copy.  The UNet generates each future frame from pure noise with
      only cross-attn fine-tuned, so fine detail present in the history never
      survives into the output.  Fix = residual parameterisation.

This discriminates them WITHOUT retraining: run the existing checkpoint under
several history spacings.  Denser spacings are out of distribution for a
stride-8 checkpoint, so they are handicapped -- if a denser history still wins,
(a) is real and worth a fine-tune.  If every spacing is equally bad, the spacing
is not the lever and (b) is where the work goes.

`copy` is the same free baseline throughout: reuse the newest history frame.

    python experiments/stage2_twophase/history_spacing.py --eps_per_task 4 --anchors 12
"""
import os, sys, pathlib, argparse, numpy as np, torch, h5py

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                       # noqa: E402
from vae_wrapper import VAEWrapper              # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/stage1_stride_k8/stage1_rollout_ep1.pt")
ap.add_argument("--K", type=int, default=8, help="must match the checkpoint's num_pred")
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--eps_per_task", type=int, default=4)
ap.add_argument("--anchors", type=int, default=12, help="anchor frames sampled per episode")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = a.device
T_H = CONFIG["num_history"]; K = a.K

# Offsets back from t, newest first.  All use exactly T_H frames, so cost is identical.
SPACINGS = {
    "stride8 (trained)": [1, 9, 17, 25, 33, 41],
    "stride4":           [1, 5,  9, 13, 17, 21],
    "stride1 (dense)":   [1, 2,  3,  4,  5,  6],
    "log-spaced":        [1, 2,  4,  8, 16, 32],
}
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

dec = lambda z: vae.decode(z.to(DEV).float()).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.
ARMS = list(SPACINGS) + ["copy"]
mse = {v: [] for v in ARMS}
geo = {v: {"rec": [], "sh": [], "miss": 0, "n": 0, "dom": []} for v in ARMS}
tdom = []
rng = np.random.default_rng(a.seed)
MAXBACK = max(max(o) for o in SPACINGS.values())
ncalls = 0

for task in a.tasks:
    p = os.path.join(REPO, "data", task, "demos_disturbed.hdf5")
    if not os.path.isfile(p):
        continue
    with h5py.File(p, "r") as f:
        eps = sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit())))
        for e in eps[:a.eps_per_task]:
            g = f[e]; n_ep = len(g["observations_cam1"])
            lo, hi = MAXBACK + 1, n_ep - K - 1
            if hi <= lo:
                continue
            # Bias toward the MIDDLE of the episode: that is where the failure is,
            # and sampling uniformly would dilute it with the calm opening.
            ts = rng.choice(np.arange(max(lo, int(0.35 * n_ep)), hi),
                            size=min(a.anchors, hi - max(lo, int(0.35 * n_ep))),
                            replace=False)
            need = sorted(set(int(t) - off for t in ts for off in range(0, MAXBACK + 1)
                              if int(t) - off >= 0))
            pos = {fr: i for i, fr in enumerate(need)}
            x = torch.from_numpy(np.array(g["observations_cam1"][need])
                                 ).float().div_(255.).permute(0, 3, 1, 2)
            act_all = torch.from_numpy(np.array(g["actions"][:n_ep], dtype=np.float32))
            with torch.no_grad():
                Zc = torch.cat([vae.encode(x[j:j + 4].to(DEV)) for j in range(0, len(x), 4)])
            for t in ts:
                t = int(t)
                truth = Zc[pos[t]].unsqueeze(0)
                im_t = dec(truth); mt = cube_masks(im_t)
                r, gg, b = im_t[..., 0], im_t[..., 1], im_t[..., 2]
                dom_t = {k: DOM[k](r, gg, b) for k in DOM}
                preds = {}
                for name, offs in SPACINGS.items():
                    hidx = [t - o for o in offs][::-1]          # oldest first
                    H = torch.stack([Zc[pos[j]] for j in hidx], 0).unsqueeze(0)
                    a_win = torch.cat([act_all[torch.tensor(hidx)],
                                       act_all[t:t + K]], 0).unsqueeze(0).to(DEV)
                    with torch.no_grad():
                        zp = cw.predict_next_latent(H, a_win, n_steps=a.ddim).float()
                    preds[name] = zp[:, 0]                      # h = 1
                    ncalls += 1
                preds["copy"] = Zc[pos[t - 1]].unsqueeze(0)
                for v, pz in preds.items():
                    mse[v].append(((pz - truth) ** 2).mean().item())
                    im = dec(pz); mr = cube_masks(im)
                    r2, g2, b2 = im[..., 0], im[..., 1], im[..., 2]
                    for k in mt:
                        A = mt[k][0]; na = int(A.sum())
                        if not (MIN_CUBE_PX <= na <= MAX_CUBE_PX):
                            continue
                        if v == ARMS[0]:
                            tdom.append(float(dom_t[k][0][A].mean()))
                        d = geo[v]; d["n"] += 1
                        d["dom"].append(float(DOM[k](r2, g2, b2)[0][A].mean()))
                        Bm = mr[k][0]; nb = int(Bm.sum())
                        d["rec"].append(nb / na)
                        if nb < 10:
                            d["miss"] += 1
                        else:
                            d["sh"].append(float(np.hypot(*(centroid(A) - centroid(Bm)))))
    print(f"  {task}: {ncalls} calls", flush=True)

td = float(np.mean(tdom))
print(f"\nh=1, mid-episode frames, {len(mse['copy'])} anchors.  truth contrast {td:.1f}")
print(f"{'history':<20} {'MSE':>8} {'recall':>7} {'shift px':>9} {'missing':>8} "
      f"{'contrast':>9} {'vs truth':>9}")
for v in ARMS:
    d = geo[v]
    sh = np.mean(d["sh"]) if d["sh"] else float("nan")
    dm = float(np.mean(d["dom"])) if d["dom"] else float("nan")
    print(f"{v:<20} {np.mean(mse[v]):8.4f} {np.mean(d['rec']):7.2f} {sh:9.2f} "
          f"{100*d['miss']/max(d['n'],1):7.1f}% {dm:9.1f} {100*dm/td:8.0f}%")
print("\nDenser spacings are OUT OF DISTRIBUTION for a stride-8 checkpoint, so they are")
print("handicapped here. If one still wins, respacing the history is worth a fine-tune.")
print("DONE")
