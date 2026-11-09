"""Compare 2-camera chunk-30 predictors on IDENTICAL held-out rollout samples.

The trainer's in-loop check uses 24 samples that differ between runs, so its numbers are noisy and not
comparable across runs.  Here every checkpoint predicts the same N (episode, t) windows drawn from the 50
held-out rollout episodes (10 per task, same split as stage1_rollout_finetune.py --holdout_per_file 10),
with both the blurry unit-noise start and the sharp sigma-scaled start.  Reports target-frame (+30) latent
MSE overall and per camera, and the paired difference vs the first checkpoint.

Checkpoint kinds are read from the checkpoint meta: plain actions, actions+pose (--pose) or pose only
(--pose_only).  Pose models are fed the TRUE pose of every history and future frame (oracle pose).

    CUDA_VISIBLE_DEVICES=0 python experiments/eval_heldout_2cam.py --n 200 --ckpts \
        warm_nopose=outputs/stage1_chunk30_2cam_ft/stage1_rollout_ep1.pt \
        old2cam=outputs/stage1_chunk30_2cam/stage1_rollout_ep3.pt \
        pose_only=outputs/stage1_chunk30_pose_only_scratch/stage1_rollout_ep1.pt
"""
import argparse, glob, os, pathlib, sys

import h5py
import numpy as np
import torch
import torch.nn.functional as Fn

REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True, help="name=path ...")
ap.add_argument("--cache", default="outputs/stage1_chunk30_2cam_ft/rollout_latents.pt")
ap.add_argument("--demos", default="data/pick_*_cube_to_tray/policy_rollouts.hdf5")
ap.add_argument("--holdout_per_file", type=int, default=10)
ap.add_argument("--split_seed", type=int, default=0, help="the trainer's --seed (split uses seed+123)")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--sample_seed", type=int, default=7)
ap.add_argument("--ddim_steps", type=int, default=10)
ap.add_argument("--out", default="outputs/eval_heldout_2cam.npz")
ap.add_argument("--modes", nargs="+", default=None, help="conditions to run (default: all for the checkpoint kind)")
ap.add_argument("--unit", action="store_true", help="also run the blurry unit-noise start for the true-condition mode")
a = ap.parse_args()

DEV = "cuda"
T_H, M, PS, CHUNK = 6, 5, 5, 30
K = CHUNK // PS
SPAN = (T_H - 1) * M + 1
CONFIG["num_pred"] = K

d = torch.load(os.path.join(REPO, a.cache), map_location="cpu")
Z = [z.float() for z in d["Z"]]; A = [x.float() for x in d["A"]]; FILE = list(d["F"])
n_ep = len(Z)

# held-out episodes: identical rule to the trainer (per-file choice with default_rng(seed+123))
rs = np.random.default_rng(a.split_seed + 123); held = set()
for fi in sorted(set(FILE)):
    idx = [e for e in range(n_ep) if FILE[e] == fi]
    held |= set(int(x) for x in rs.choice(idx, min(a.holdout_per_file, len(idx)), replace=False))
held = sorted(held); print(f"{len(held)} held-out episodes")

# raw poses (ee pos 3 + joint pos 9), same episode order as the cache
P_RAW = []
for p_ in sorted(glob.glob(os.path.join(REPO, a.demos))):
    with h5py.File(p_, "r") as f_:
        for e_ in list(f_):
            if "observations_cam1" in f_[e_]:
                g_ = f_[e_]
                P_RAW.append(torch.from_numpy(np.concatenate(
                    [np.array(g_["robot_ee_pos"]), np.array(g_["robot_joint_pos"])], 1)).float())
assert len(P_RAW) == n_ep, (len(P_RAW), n_ep)

# the fixed evaluation windows
rng = np.random.default_rng(a.sample_seed)
samples = []
for _ in range(a.n):
    e = int(held[rng.integers(len(held))]); t = int(rng.integers(SPAN, len(Z[e]) - CHUNK + 1))
    samples.append((e, t))


def hidx(t):
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]


def fut_idx(t):
    return [t - 1 + PS * j for j in range(1, K + 1)]


def grad_energy(x, m):
    """mean |finite-difference gradient| of a latent (C,H,W) inside mask m (H,W): a smear/sharpness proxy."""
    gx = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(0); gy = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(0)
    mx = m[:, 1:] * m[:, :-1]; my = m[1:, :] * m[:-1, :]
    return ((gx * mx).sum() + (gy * my).sum()) / (mx.sum() + my.sum()).clamp(min=1)


def acvec(e, t, meta, mode="true", other=None):
    """Same conditioning vector the trainer builds for this checkpoint kind.
    mode: true | zero (all-zero conditioning) | shuffle (conditioning of another window) | hold (pose models:
    future poses replaced by the last history pose, i.e. no knowledge of the motion)."""
    if mode == "shuffle":
        e, t = other
    fa = torch.stack([A[e][t + PS * (j - 1): t + PS * j].mean(0) for j in range(1, K + 1)])
    ac = torch.cat([A[e][hidx(t)], fa], 0)
    if not meta.get("pose"):
        return torch.zeros_like(ac) if mode == "zero" else ac
    mean = torch.tensor(meta["pose_mean"]); std = torch.tensor(meta["pose_std"]); sc = meta.get("pose_scale", 1.0)
    hi = hidx(t)
    px = P_RAW[e][hi + fut_idx(t)]
    if mode == "hold":
        px = torch.cat([px[:T_H], P_RAW[e][hi[-1]].unsqueeze(0).expand(K, -1)], 0)
    pe = torch.cat([(px - mean) / std * sc, torch.ones(len(px), 1)], 1)
    if mode == "zero":
        pe = torch.zeros_like(pe)
    return pe if meta.get("pose_only") else torch.cat([ac, pe], 1)



rng2 = np.random.default_rng(a.sample_seed + 1)
others = [samples[int(rng2.integers(len(samples)))] for _ in samples]
for spec in a.ckpts:
    name, path = spec.split("=", 1)
    ck = torch.load(os.path.join(REPO, path), map_location="cpu")
    meta = ck.get("meta", {})
    adim = (meta["pose_dim"] if meta.get("pose_only") else CONFIG["action_dim"] + (meta.get("pose_dim") or 0)) \
        if meta.get("pose") else CONFIG["action_dim"]
    cw = CtrlWorldWrapper(action_dim=adim, num_history=T_H, num_pred=K, svd_path=CONFIG["svd_model_name"],
                          freeze_unet=True, finetune_cross_attn=True, extra_unfreeze=tuple(meta.get("unfreeze", [])),
                          dtype=torch.float16).to(DEV)
    cw.action_encoder.load_state_dict(ck["action_encoder_state"])
    cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
    if "unet_extra_state" in ck:
        cw.load_unet_extra_state_dict(ck["unet_extra_state"])
    cw.eval()
    modes = a.modes or (["true", "zero", "shuffle"] + (["hold"] if meta.get("pose") else []))
    print(f"[{name}] epoch {ck.get('epoch')}  action_dim {adim}  pose={meta.get('pose')} only={meta.get('pose_only')} modes={modes}", flush=True)
    out = {m: [] for m in modes}; outb = {m: [] for m in modes}; outw = {m: [] for m in modes}
    unit, copy = [], []
    reg, shp = [], []
    for i, (e, t) in enumerate(samples):
        H = Z[e][hidx(t)].unsqueeze(0).to(DEV)
        tg = torch.stack([Z[e][j] for j in fut_idx(t)]).to(DEV)                # (K,4,h,w)
        W = tg.shape[-1] // 2
        with torch.no_grad():
            for m in modes:
                ac = acvec(e, t, meta, m, others[i]).unsqueeze(0).to(DEV)
                torch.manual_seed(a.sample_seed * 1000 + i)
                p = cw.predict_next_latent(H, ac, n_steps=a.ddim_steps, init_scale=True).float()[0]
                out[m].append([Fn.mse_loss(p[j], tg[j]).item() for j in range(K)])
                outb[m].append(Fn.mse_loss(p[-1][..., :W], tg[-1][..., :W]).item())
                outw[m].append(Fn.mse_loss(p[-1][..., W:], tg[-1][..., W:]).item())
                if m == "true":                       # moved-region error and sharpness of the +30 prediction
                    mv = (((tg[-1] - H[0, -1]) ** 2).mean(0) > 0.05).float()
                    nmv = mv.sum().item()
                    if nmv >= 20:
                        reg.append(((p[-1] - tg[-1]) ** 2).mean(0).mul(mv).sum().item() / nmv)
                        shp.append((grad_energy(p[-1], mv) / grad_energy(tg[-1], mv).clamp(min=1e-6)).item())
                    else:
                        reg.append(np.nan); shp.append(np.nan)
                if m == "true" and a.unit:
                    torch.manual_seed(a.sample_seed * 1000 + i)
                    pu = cw.predict_next_latent(H, ac, n_steps=a.ddim_steps).float()[0, -1]
                    unit.append(Fn.mse_loss(pu, tg[-1]).item())
        copy.append([Fn.mse_loss(H[0, -1], tg[j]).item() for j in range(K)])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{a.n}", flush=True)
    se = lambda x: x.std() / np.sqrt(len(x))
    print(f"\n[{name}] sharp-start latent MSE per horizon (mean over {a.n} identical held-out windows)")
    print(f"{'mode':9s}" + "".join(f"{'+' + str(PS * (j + 1)):>8s}" for j in range(K)) + "   +30 base  +30 wrist   +30 +-sem")
    for m in modes:
        o = np.array(out[m])
        print(f"{m:9s}" + "".join(f"{o[:, j].mean():8.4f}" for j in range(K)) +
              f"   {np.mean(outb[m]):8.4f} {np.mean(outw[m]):9.4f}   {se(o[:, -1]):.4f}")
        if m != "true":
            dd = o[:, -1] - np.array(out["true"])[:, -1]
            print(f"{'':9s}paired (this - true) at +30: {dd.mean():+.4f} +- {se(dd):.4f}")
    c = np.array(copy); print(f"{'copy-last':9s}" + "".join(f"{c[:, j].mean():8.4f}" for j in range(K)))
    if unit: print(f"unit-start +30: {np.mean(unit):.4f}")
    print(f"moved-region +30 MSE: {np.nanmean(reg):.4f}   sharpness ratio (pred/true gradient energy in moved region, 1 = as sharp as truth): {np.nanmean(shp):.3f}")
    os.makedirs(os.path.join(REPO, os.path.dirname(a.out)), exist_ok=True)
    np.savez(os.path.join(REPO, a.out.replace(".npz", f"_{name}.npz")), samples=np.array(samples), copy=c, unit=np.array(unit), moved_mse=np.array(reg), sharp_ratio=np.array(shp),
             **{f"{m}": np.array(out[m]) for m in modes}, **{f"{m}_base": np.array(outb[m]) for m in modes},
             **{f"{m}_wrist": np.array(outw[m]) for m in modes})
    del cw; torch.cuda.empty_cache()
print("DONE")
