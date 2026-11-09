"""Stage A for the Pi0Fast scoring of held-out ROLLOUT windows, for any mix of action / pose-only / unfrozen checkpoints.
Builds, per window, the +30 base+wrist images of: real (true frames), vae (round trip of the true latents), copy (last history
frame, stale) and one arm per checkpoint (sharp start).  Output is the 'NEW' format that score_arms_pi0fast.py reads
(img_<arm>/wimg_<arm>, real, wrist, task, demo_a, n_obj).  Pose models are fed the TRUE pose (oracle).
    python experiments/make_arms_rollouts.py --n 100 --ckpts warm_action=... pose_ep3=... --out_npz outputs/arms_rollouts.npz"""
import argparse, glob, json, os, pathlib, sys
import h5py, numpy as np, torch
REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from config import CONFIG
from ctrl_world_wrapper import CtrlWorldWrapper
from vae_wrapper import VAEWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--cache", default="outputs/stage1_chunk30_2cam_ft/rollout_latents.pt")
ap.add_argument("--demos", default="data/pick_*_cube_to_tray/policy_rollouts.hdf5")
ap.add_argument("--holdout_per_file", type=int, default=10)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--out_npz", required=True)
a = ap.parse_args()
DEV = "cuda"; T_H, M, PS, CHUNK = 6, 5, 5, 30; K = CHUNK // PS; SPAN = (T_H - 1) * M + 1
CONFIG["num_pred"] = K
d = torch.load(os.path.join(REPO, a.cache), map_location="cpu")
Z = [z.float() for z in d["Z"]]; A = [x.float() for x in d["A"]]; FILE = list(d["F"]); n_ep = len(Z)
rs = np.random.default_rng(123); held = set()
for fi in sorted(set(FILE)):
    idx = [e for e in range(n_ep) if FILE[e] == fi]
    held |= set(int(x) for x in rs.choice(idx, min(a.holdout_per_file, len(idx)), replace=False))
held = sorted(held)
index, P_RAW = [], []
for p_ in sorted(glob.glob(os.path.join(REPO, a.demos))):
    with h5py.File(p_, "r") as f_:
        for k_ in list(f_):
            if "observations_cam1" in f_[k_]:
                g_ = f_[k_]; index.append((p_, k_))
                P_RAW.append(torch.from_numpy(np.concatenate([np.array(g_["robot_ee_pos"]), np.array(g_["robot_joint_pos"])], 1)).float())
assert len(index) == n_ep
hidx = lambda t: [max(0, t - 1 - i * M) for i in range(T_H)][::-1]
fidx = lambda t: [t - 1 + PS * j for j in range(1, K + 1)]
rng = np.random.default_rng(a.seed); samples = []
for _ in range(a.n):
    e = int(held[rng.integers(len(held))]); t = int(rng.integers(SPAN, len(Z[e]) - 2 * CHUNK + 1)); samples.append((e, t))

def cond(e, t, meta):
    fa = torch.stack([A[e][t + PS * (j - 1): t + PS * j].mean(0) for j in range(1, K + 1)])
    ac = torch.cat([A[e][hidx(t)], fa], 0)
    if not meta.get("pose"): return ac
    mean = torch.tensor(meta["pose_mean"]); std = torch.tensor(meta["pose_std"]); sc = meta.get("pose_scale", 1.0)
    px = P_RAW[e][hidx(t) + fidx(t)]
    pe = torch.cat([(px - mean) / std * sc, torch.ones(len(px), 1)], 1)
    return pe if meta.get("pose_only") else torch.cat([ac, pe], 1)

preds, names = {}, []
for spec in a.ckpts:
    name, path = spec.split("=", 1)
    ck = torch.load(os.path.join(REPO, path), map_location="cpu"); meta = ck.get("meta", {})
    adim = (meta["pose_dim"] if meta.get("pose_only") else CONFIG["action_dim"] + (meta.get("pose_dim") or 0)) if meta.get("pose") else CONFIG["action_dim"]
    cw = CtrlWorldWrapper(action_dim=adim, num_history=T_H, num_pred=K, svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                          finetune_cross_attn=True, extra_unfreeze=tuple(meta.get("unfreeze", [])), dtype=torch.float16).to(DEV)
    cw.action_encoder.load_state_dict(ck["action_encoder_state"]); cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
    if "unet_extra_state" in ck: cw.load_unet_extra_state_dict(ck["unet_extra_state"])
    cw.eval(); out = []
    for i, (e, t) in enumerate(samples):
        H = Z[e][hidx(t)].unsqueeze(0).to(DEV)
        with torch.no_grad():
            torch.manual_seed(a.seed * 1000 + i)
            out.append(cw.predict_next_latent(H, cond(e, t, meta).unsqueeze(0).to(DEV), n_steps=10, init_scale=True).float()[0, -1].cpu())
        if (i + 1) % 25 == 0: print(f"  [{name}] {i + 1}/{a.n}", flush=True)
    preds[name] = out; names.append(name); del cw; torch.cuda.empty_cache(); print("done", name, flush=True)

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV)
u8 = lambda x: (x.permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
out = {k: [] for k in ["real", "wrist", "task", "demo_a", "n_obj", "ep", "t"]}
for k in ["vae", "copy"] + names: out["img_" + k] = []; out["wimg_" + k] = []
instr = {}
for i, (e, t) in enumerate(samples):
    f_tgt = fidx(t)[-1]
    lat = {"vae": Z[e][f_tgt], "copy": Z[e][hidx(t)[-1]], **{n: preds[n][i] for n in names}}
    with torch.no_grad(): dec = vae.decode(torch.stack([lat[k] for k in lat]).to(DEV)).cpu()
    W = dec.shape[3] // 2
    for j, k in enumerate(lat):
        out["img_" + k].append(u8(dec[j][:, :, :W])); out["wimg_" + k].append(u8(dec[j][:, :, W:]))
    p, key = index[e]
    with h5py.File(p, "r") as f:
        out["real"].append(np.array(f[key]["observations_cam1"][f_tgt])); out["wrist"].append(np.array(f[key]["observations_cam2"][f_tgt]))
    td = os.path.dirname(p)
    if td not in instr: instr[td] = json.load(open(os.path.join(td, "task.json")))["instruction"]
    out["task"].append(instr[td]); out["demo_a"].append(A[e][f_tgt:f_tgt + CHUNK].numpy())
    out["n_obj"].append(int((((Z[e][f_tgt] - Z[e][hidx(t)[-1]]) ** 2).mean(0) > 0.05).sum())); out["ep"].append(e); out["t"].append(t)
np.savez_compressed(os.path.join(REPO, a.out_npz), **{k: np.array(v) for k, v in out.items()}, ckpt=",".join(a.ckpts))
print("saved", a.out_npz, "DONE")
