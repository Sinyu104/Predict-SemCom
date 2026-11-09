"""Fine-tune Stage-1 Ctrl-World on CLOSED-LOOP rollout history.

Stage 1 was trained on ground-truth history: at deployment it consumes its own
reconstructions, whose error is spatially smooth and temporally correlated, and
it amplifies that error (~2x per cycle) until the loop collapses to copying its
own anchor.  Nothing in the original training loop exposes this, because both
train and val run on clean history.

This fine-tunes on the histories the receiver loop ACTUALLY produces, collected
by experiments/rollout_collect.py:

    history = ZR[t-num_history : t]     rollout reconstructions (drifted)
    action  = A [t-num_history : t+1]
    target  = ZT[t]                     the TRUE latent

The clip handed to forward_ddpm is cat([drifted history, true target]), so the
existing objective is reused unchanged — only the history distribution differs.
forward_ddpm still applies its own sigma_h augmentation on top; that is left
alone, since sigma_h ~ N(0, 0.3) is centred at zero and many samples therefore
see the drifted history essentially as-is.

Reports the metric that actually matters — AMPLIFICATION: inject a known error
into clean history and measure how much of it survives into the prediction.
Below ~1 the closed loop is stable.  Training loss alone will not tell you this.

    python experiments/stage1_rollout_finetune.py \
        --shards "outputs/rollout/shard*.pt" --epochs 3 \
        --out outputs/stage1_rollout_ft

CHUNK MODE (--chunk 30, for PI0Fast's 30-step action chunk)
-----------------------------------------------------------
The receiver gets one frame per chunk, so the predictor must jump a whole chunk:
given strided history ending at frame t-1 and the chunk's actions a_t..a_{t+29},
predict frame t+29.  Only that last frame is the target (weight 1); the
intermediate frames at every --pred_stride steps are also generated, because they
are fed back as history, but get only a weak --aux_w loss so they stay plausible.
Future slot j (frame t-1+pred_stride*j) is conditioned on the MEAN of the
pred_stride actions leading to it.  Rollout shards are only ~30 frames long, so
chunk mode reads full episodes from --demos (VAE-encoded once, cached) and
simulates receiver drift with coherent blur on the history (--drift_p/--drift_max).

    python experiments/stage1_rollout_finetune.py --chunk 30 --pred_stride 5 \
        --history_stride 10 --stage1_ckpt outputs/stage1_rollout_ft/stage1_rollout_ep1.pt \
        --out outputs/stage1_chunk30
"""
import argparse, glob, os, pathlib, sys, time

import numpy as np
import torch
import torch.nn.functional as Fn

REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from config import CONFIG                        # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=str, default="outputs/rollout/shard*.pt")
ap.add_argument("--out", type=str, default="outputs/stage1_rollout_ft")
ap.add_argument("--stage1_ckpt", type=str,
                default="outputs/stage1_5cube_cam1_K8/stage1_best.pt")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--val_frac", type=float, default=0.1)
ap.add_argument("--max_snr", type=float, default=None,
                help="Drop episodes above this SNR (drift turned out flat in SNR, "
                     "so this is mostly a filtering hook rather than a needed knob).")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--history_stride", type=int, default=1,
                help="m: must match the stride used to COLLECT the rollout data, and "
                     "the stride used at inference. Read it from the shard meta.")
ap.add_argument("--num_pred", type=int, default=1,
                help="K: must match the checkpoint and the rollout data. Clip fed to "
                     "forward_ddpm is [drifted history (num_history) | true targets (K)].")
ap.add_argument("--chunk", type=int, default=0,
                help="Chunk mode: predict the frame --chunk steps ahead (30 for PI0Fast). 0 = legacy.")
ap.add_argument("--pred_stride", type=int, default=5,
                help="Chunk mode: spacing of the generated future frames; K = chunk/pred_stride. "
                     "Only the last is the target, the rest are weakly supervised intermediates.")
ap.add_argument("--aux_w", type=float, default=0.2, help="Chunk mode: loss weight of intermediate frames.")
ap.add_argument("--demos", type=str, default="data/*/demos*.hdf5", help="Chunk mode: demo files.")
ap.add_argument("--max_eps_per_file", type=int, default=None)
ap.add_argument("--latent_cache", type=str, default="outputs/stage1_chunk30/latents.pt")
ap.add_argument("--max_val", type=int, default=512,
                help="Cap on val samples (eval runs on rank 0 while other ranks wait).")
ap.add_argument("--drift_p", type=float, default=0.5,
                help="Chunk mode: prob. a sample's history gets coherent-blur drift.")
ap.add_argument("--drift_max", type=float, default=0.05, help="Chunk mode: max injected history MSE.")
ap.add_argument("--views", type=int, default=1,
                help="Chunk mode: 2 = predict base (cam1) AND wrist (cam2) jointly; the two latents are "
                     "concatenated along width (4x28x56), the Ctrl-World multi-view layout.")
ap.add_argument("--cam1_cache", type=str, default="",
                help="views=2: existing cam1 latent cache to reuse, so only cam2 is encoded.")
ap.add_argument("--extra_pose", action="store_true",
                help="The --extra_cache files carry pose (key P, e.g. outputs/rollouts2/latents_*.pt): use it (needed by --pose_only).")
ap.add_argument("--extra_holdout", type=int, default=0,
                help="With --extra_pose: hold out this many whole episodes of EACH extra cache for validation.")
ap.add_argument("--extra_cache", type=str, nargs="*", default=[],
                help="Existing latent caches added to the TRAIN set only (e.g. the demo latents).")
ap.add_argument("--holdout_per_file", type=int, default=0,
                help="Hold out this many whole episodes per --demos file for validation "
                     "(true held-out; train never sees them). 0 = random sample split.")
ap.add_argument("--skip_before", action="store_true", help="Skip the initial eval (already measured).")
ap.add_argument("--pose", action="store_true",
                help="Condition on robot pose (ee pos + joint pos): EVERY frame, history and future slot, gets its own "
                     "TRUE pose (the client reports the poses after executing the 30 steps). This is an oracle-pose "
                     "setting. Demo episodes (no pose recorded) get zeros + a missing flag.")
ap.add_argument("--pose_scale", type=float, default=1.0,
                help="Multiply the normalised pose inputs (not the flag) by this: with Adam it is equivalent to a "
                     "pose_scale x larger learning rate on the pose columns of the first action-encoder layer.")
ap.add_argument("--pose_only", action="store_true",
                help="Ctrl-World style: condition on pose ONLY (no action inputs). Implies --pose. The first layer of the "
                     "action encoder is re-initialised (input size changes); later layers and the UNet cross-attn are kept. "
                     "Needs pose for every frame, so train on rollouts only (no --extra_cache).")
ap.add_argument("--first_layer_lr", type=float, default=1e-4,
                help="--pose_only: learning rate of the freshly initialised first layer (rest uses --lr).")
ap.add_argument("--extra_lr", type=float, default=None, help="learning rate for the --unfreeze parameters (default: --lr)")
ap.add_argument("--unfreeze", nargs="*", default=[], help="regexes on UNet parameter names to train in addition to attn2 and the action encoder, e.g. '\\.attn1\\.' 'temporal_res_block'")
ap.add_argument("--scratch", action="store_true",
                help="Do not load --stage1_ckpt: the action encoder starts random and the UNet cross-attn at the "
                     "pretrained SVD weights (the original Ctrl-World recipe). With --pose_only the whole action "
                     "encoder trains at --first_layer_lr.")
ap.add_argument("--epoch_samples", type=int, default=0,
                help="If >0, each epoch uses a random subset of this many train samples "
                     "(neighbouring t are near-duplicates).")
ap.add_argument("--recon_hist", type=str, default="",
                help="Glob of make_recon_history_2cam.py shards: history = codec-reconstructed frames (z~) built from clean history; "
                     "targets stay true. Only t = 1 (mod 5) windows of the cached episodes are used.")
ap.add_argument("--clean_p", type=float, default=0.5, help="With --recon_hist: prob. of a true-history sample (training only).")
a = ap.parse_args()

# Data parallel (launch with python -m torch.distributed.run --nproc_per_node=N).
# Only ~100M params are trainable, so gradients are averaged by hand instead of
# wrapping in DDP (frozen UNet + gradient checkpointing fight DDP's hooks).
import torch.distributed as dist
WORLD = int(os.environ.get("WORLD_SIZE", 1)); RANK = int(os.environ.get("RANK", 0))
if WORLD > 1:
    import datetime
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
    torch.manual_seed(a.seed + RANK); np.random.seed(a.seed + RANK)   # per-rank noise/drift
    DEV = f"cuda:{int(os.environ['LOCAL_RANK'])}"
else:
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
if RANK:
    print = lambda *x, **k: None            # log from rank 0 only
T_H  = CONFIG["num_history"]
CHUNK = a.chunk
PS   = a.pred_stride
if CHUNK:
    assert CHUNK % PS == 0, "--chunk must be a multiple of --pred_stride"
    a.num_pred = CHUNK // PS
K    = a.num_pred
M    = a.history_stride
FW = None
if CHUNK:                                   # weight: last frame 1, intermediates aux_w
    FW = torch.full((K,), a.aux_w); FW[-1] = 1.0
SPAN = (T_H - 1) * M + 1          # frames the strided window reaches back over
CONFIG["num_pred"] = K
os.makedirs(a.out, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────── #
# Per-episode lists (episodes may differ in length in chunk mode).
def load_shards():
    ZT, ZR, A, SNR = [], [], [], []
    for f in sorted(glob.glob(a.shards)):
        d = torch.load(f, map_location="cpu")
        for i in range(len(d["ZT"])):
            ZT.append(d["ZT"][i].float()); ZR.append(d["ZR"][i].float()); A.append(d["A"][i].float())
        SNR.append(d.get("SNR", torch.full((len(d["ZT"]),), float("nan"))))
        print(f"  loaded {f}: {tuple(d['ZT'].shape)}")
    return ZT, ZR, A, torch.cat(SNR)

def load_demos():
    """Encode full episodes once (cam1 [+ cam2 -> width concat]), cache as fp16.
    Encoding is sharded over ranks (item i -> rank i % WORLD), then merged by rank 0."""
    if WORLD > 1: dist.barrier()                       # nobody races the existence check
    if os.path.exists(a.latent_cache):
        d = torch.load(a.latent_cache, map_location="cpu")
        print(f"  loaded latent cache {a.latent_cache}: {len(d['Z'])} episodes")
        return [z.float() for z in d["Z"]], [x.float() for x in d["A"]], list(d.get("F", [0] * len(d["Z"])))
    import h5py
    from vae_wrapper import VAEWrapper
    items = []
    for fi, p in enumerate(sorted(glob.glob(os.path.join(REPO, a.demos)))):
        with h5py.File(p, "r") as f:
            items += [(fi, p, e) for e in list(f)[:a.max_eps_per_file] if "observations_cam1" in f[e]]
    print(f"  {len(items)} episodes to encode, {WORLD} rank(s)", flush=True)
    vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV)
    old = None
    if a.views == 2 and a.cam1_cache and os.path.exists(a.cam1_cache):
        old = torch.load(a.cam1_cache, map_location="cpu")["Z"]; print(f"  reusing cam1 latents ({len(old)} eps)")
    mine = {"Z": [], "A": [], "F": [], "gi": []}
    for gi in range(RANK, len(items), WORLD):
        fi, p, e = items[gi]
        with h5py.File(p, "r") as f:
            g = f[e]
            def enc(name):
                x = torch.from_numpy(np.array(g[name])).float().div_(255.).permute(0, 3, 1, 2)
                with torch.no_grad():
                    return torch.cat([vae.encode(x[j:j + 16].to(DEV)).cpu() for j in range(0, len(x), 16)])
            if a.views == 2:
                z1 = old[gi].float() if old is not None else enc("observations_cam1")
                z = torch.cat([z1, enc("observations_cam2")], dim=-1)          # width concat
            else:
                z = enc("observations_cam1")
            mine["Z"].append(z.half()); mine["A"].append(torch.from_numpy(np.array(g["actions"], dtype=np.float32)).half())
            mine["F"].append(fi); mine["gi"].append(gi)
        if len(mine["Z"]) % 10 == 0:
            print(f"  rank{RANK} encoded {len(mine['Z'])}/{len(range(RANK, len(items), WORLD))}", flush=True)
    del vae; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.latent_cache) or ".", exist_ok=True)
    torch.save(mine, f"{a.latent_cache}.r{RANK}")
    if WORLD > 1: dist.barrier()
    if RANK == 0:
        parts = [torch.load(f"{a.latent_cache}.r{r}", map_location="cpu") for r in range(WORLD)]
        order = sorted((gi, r, j) for r, pt in enumerate(parts) for j, gi in enumerate(pt["gi"]))
        out = {"Z": [parts[r]["Z"][j] for _, r, j in order], "A": [parts[r]["A"][j] for _, r, j in order],
               "F": [parts[r]["F"][j] for _, r, j in order]}
        torch.save(out, a.latent_cache)
        for r in range(WORLD): os.remove(f"{a.latent_cache}.r{r}")
    if WORLD > 1: dist.barrier()
    d = torch.load(a.latent_cache, map_location="cpu")
    return [z.float() for z in d["Z"]], [x.float() for x in d["A"]], list(d["F"])

if CHUNK:
    ZT, A, FILE = load_demos(); ZR = ZT                       # drift is simulated per batch
    N_MAIN = len(ZT)
    P_EXTRA = []
    for k_, pth in enumerate(a.extra_cache):                  # train-only extra episodes
        d_ = torch.load(pth, map_location="cpu")
        ZT += [z.float() for z in d_["Z"]]; A += [x.float() for x in d_["A"]]
        FILE += [1000 + k_ if (a.extra_pose and a.extra_holdout) else -1] * len(d_["Z"])
        if a.extra_pose: P_EXTRA += [x.float() for x in d_["P"]]
        print(f"  + {len(d_['Z'])} train-only episodes from {pth}")
    SNR = torch.full((len(ZT),), float("nan"))
else:
    ZT, ZR, A, SNR = load_shards()
    if a.max_snr is not None:
        keep = [i for i in range(len(ZT)) if SNR[i] <= a.max_snr]
        ZT, ZR, A, SNR = [ZT[i] for i in keep], [ZR[i] for i in keep], [A[i] for i in keep], SNR[keep]
n_ep = len(ZT)
T = [len(z) for z in ZT]
# last usable t: legacy target t..t+K-1 ; chunk target frame t-1+CHUNK
t_hi = (lambda n: n - CHUNK) if CHUNK else (lambda n: n - K)
err0 = np.mean([((ZR[e][SPAN:] - ZT[e][SPAN:]) ** 2).mean().item() for e in range(n_ep)])
print(f"[data] {n_ep} episodes (mean {np.mean(T):.0f} frames)  K={K} stride={M} chunk={CHUNK} "
      f"(span={(T_H-1)*M})   mean history error={err0:.4f}")

samples = [(e, t) for e in range(n_ep) for t in range(SPAN, t_hi(T[e]) + 1)]
if a.recon_hist:
    ZR = list(ZT); have = set()
    for fn in sorted(glob.glob(a.recon_hist)):
        d = torch.load(fn, map_location="cpu")
        for e in sorted(set(d["e"].tolist())):
            if e not in have: ZR[e] = ZT[e].clone(); have.add(e)
        for j in range(len(d["e"])):
            ZR[int(d["e"][j])][int(d["f"][j])] = d["zr"][j].float()
    lo = SPAN + CHUNK - 1 + (T_H - 1) * M + 1           # oldest history frame must itself be a reconstruction
    samples = [(e, t) for e, t in samples if e in have and t % M == 1 and t >= lo]
    print(f"[recon_hist] {len(have)} episodes, {len(samples)} windows; history err "
          f"{np.mean([((ZR[e][SPAN+CHUNK-1:] - ZT[e][SPAN+CHUNK-1:]) ** 2).mean().item() for e in have]):.4f} (over all frames; "
          f"only grid frames are reconstructed)", flush=True)
rng = np.random.default_rng(a.seed); rng.shuffle(samples)
if CHUNK and a.holdout_per_file:
    _r = np.random.default_rng(a.seed + 123); held = set()
    for fi in sorted(set(f_ for f_ in FILE if f_ >= 0)):
        _idx = [e for e in range(len(ZT)) if FILE[e] == fi]
        _n = a.extra_holdout if fi >= 1000 else a.holdout_per_file
        held |= set(int(x) for x in _r.choice(_idx, min(_n, len(_idx)), replace=False))
    val_all = [x for x in samples if x[0] in held]
    val_s = val_all[:a.max_val]; train_s = [x for x in samples if x[0] not in held]
    print(f"[data] held out {len(held)} whole episodes ({len(val_all)} samples, {len(val_s)} used for val)")
else:
    n_val = max(1, int(len(samples) * a.val_frac))
    val_s, train_s = samples[:n_val][:a.max_val], samples[n_val:]
print(f"[data] {len(train_s)} train / {len(val_s)} val samples")

POSE = a.pose or a.pose_only
POSE_ONLY = a.pose_only
if POSE_ONLY:
    assert not a.extra_cache or a.extra_pose, "--pose_only needs pose for every episode: do not pass --extra_cache (or use --extra_pose)"
NP = 0
if POSE:
    assert CHUNK, "--pose is implemented for chunk mode"
    import h5py
    def load_pose():
        """Pose per frame (ee pos 3 + joint pos 9), same episode order as load_demos()."""
        out = []
        for p_ in sorted(glob.glob(os.path.join(REPO, a.demos))):
            with h5py.File(p_, "r") as f_:
                for e_ in list(f_)[:a.max_eps_per_file]:
                    if "observations_cam1" not in f_[e_]:
                        continue
                    g_ = f_[e_]
                    out.append(torch.from_numpy(np.concatenate(
                        [np.array(g_["robot_ee_pos"]), np.array(g_["robot_joint_pos"])], 1)).float())
        return out
    P = load_pose() + P_EXTRA
    NPOSE = len(P)
    assert NPOSE == N_MAIN + len(P_EXTRA), (NPOSE, N_MAIN)
    assert all(len(P[i]) == len(ZT[i]) for i in range(NPOSE)), "pose/frame count mismatch"
    cat_ = torch.cat(P[:N_MAIN]); P_MEAN, P_STD = cat_.mean(0), cat_.std(0).clamp_min(1e-3)
    def _norm(x):
        return torch.cat([(x - P_MEAN) / P_STD * a.pose_scale, torch.ones(len(x), 1)], 1)   # + "pose valid" flag
    P = [_norm(x) for x in P] + [torch.zeros(len(ZT[i]), 13) for i in range(NPOSE, len(ZT))]   # extras without pose: zeros, flag 0
    NP = P[0].shape[1]
    print(f"[pose] {NPOSE} episodes with pose, {len(ZT) - NPOSE} without; pose features = {NP}", flush=True)

def acvec(e, t):
    """Per-frame conditioning: (T_H + K, 7 [+ NP]).  Every frame, history AND future slot, carries its own TRUE pose:
    the client reports the poses of the 30 steps after executing them, so the receiver has them when it predicts."""
    if POSE_ONLY:
        return P[e][hidx(t) + fut_idx(t)]
    ac_ = torch.cat([A[e][hidx(t)], fut_actions(e, t)], 0)
    if not POSE:
        return ac_
    return torch.cat([ac_, P[e][hidx(t) + fut_idx(t)]], 1)

def hidx(t):
    """Strided history indices ending at t-1, oldest first — matches collection."""
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]

def fut_idx(t):
    """Frame indices of the K generated future frames (last one = the target)."""
    return [t + i for i in range(K)] if not CHUNK else [t - 1 + PS * j for j in range(1, K + 1)]

def fut_actions(e, t):
    if not CHUNK:
        return A[e][t:t + K]
    return torch.stack([A[e][t + PS * (j - 1): t + PS * j].mean(0) for j in range(1, K + 1)])

def drift(h):
    """Coherent blur of strength ~U(0, drift_max) MSE, same blur on all history frames."""
    if not CHUNK or np.random.rand() > a.drift_p:
        return h
    inj = np.random.rand() * a.drift_max
    hb = blur(h.unsqueeze(0))[0]
    base = Fn.mse_loss(hb, h).item()
    return h + (inj / max(base, 1e-12)) ** 0.5 * (hb - h)

def batch(idx, train=True):
    """clip = [strided drifted history (T_H) | future frames (K)]."""
    def src(e):
        return ZT[e] if (a.recon_hist and train and np.random.rand() < a.clean_p) else ZR[e]
    zc = torch.stack([torch.cat([drift(src(e)[hidx(t)]), ZT[e][fut_idx(t)]], 0) for e, t in idx])
    ac = torch.stack([acvec(e, t) for e, t in idx])
    return zc.to(DEV), ac.to(DEV)

# ── model ─────────────────────────────────────────────────────────────────── #
cw = CtrlWorldWrapper(
    action_dim=(NP if POSE_ONLY else CONFIG["action_dim"] + NP), num_history=T_H, num_pred=K,
    svd_path=CONFIG["svd_model_name"], freeze_unet=True,
    finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False), extra_unfreeze=tuple(a.unfreeze),
    dtype=torch.float16).to(DEV)
ck = None if a.scratch else torch.load(a.stage1_ckpt, map_location="cpu")
if ck is not None:
    _sd = ck["action_encoder_state"]
    if POSE_ONLY and not ck.get("meta", {}).get("pose_only"):   # input size changes: keep layers 2-3, re-init layer 1
        _sd = {k: v for k, v in _sd.items() if not k.startswith("net.0.")}
    if NP and not POSE_ONLY and _sd["net.0.weight"].shape[1] == CONFIG["action_dim"]:      # widen: new pose inputs start at zero weight
        _w = _sd["net.0.weight"]; _sd["net.0.weight"] = torch.cat([_w, torch.zeros(_w.shape[0], NP)], 1)
    cw.action_encoder.load_state_dict(_sd, strict=False)
    cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
    if "unet_extra_state" in ck:
        cw.load_unet_extra_state_dict(ck["unet_extra_state"])
else:
    print("[scratch] action encoder random, UNet cross-attn = pretrained SVD (no Stage-1 checkpoint loaded)", flush=True)
del ck
params = list(cw.trainable_parameters())
_xid = {id(q) for q in cw.extra_parameters()}
_groups = []
if POSE_ONLY:
    _first = list((cw.action_encoder if a.scratch else cw.action_encoder.net[0]).parameters()); _fid = {id(q) for q in _first}
    _groups.append({"params": _first, "lr": a.first_layer_lr})
else:
    _fid = set()
_groups.append({"params": [q for q in params if id(q) in _xid], "lr": a.extra_lr if a.extra_lr is not None else a.lr})
_groups.append({"params": [q for q in params if id(q) not in _xid and id(q) not in _fid], "lr": a.lr})
opt = torch.optim.AdamW([g for g in _groups if g["params"]])
scaler = torch.amp.GradScaler("cuda")
print(f"[model] {sum(p.numel() for p in params):,} trainable params  lr={a.lr}")

# ── amplification metric ──────────────────────────────────────────────────── #
def blur(h):
    B, Tn, C, H, W = h.shape
    return Fn.avg_pool2d(h.reshape(-1, C, H, W), 3, 1, 1).reshape(B, Tn, C, H, W)

@torch.no_grad()
def amplification(n_probe=6, inj=0.02):
    """Inject `inj` MSE of COHERENT blur into clean history; report
    (excess prediction error) / (injected error).  <1 means a stable loop."""
    cw.eval(); out = []
    for k in range(n_probe):
        e, t = val_s[k]
        H0 = ZT[e][hidx(t)].unsqueeze(0).to(DEV)
        ac = acvec(e, t).unsqueeze(0).to(DEV)
        tgt = ZT[e][fut_idx(t)[-1]].unsqueeze(0).to(DEV)          # the target frame
        clean = Fn.mse_loss(cw.predict_next_latent(H0, ac, n_steps=10)[:, -1].float(), tgt)
        Hb = blur(H0); base = Fn.mse_loss(Hb, H0).item()
        Hc = H0 + (inj / max(base, 1e-12)) ** 0.5 * (Hb - H0)     # same blur all frames
        dirty = Fn.mse_loss(cw.predict_next_latent(Hc, ac, n_steps=10)[:, -1].float(), tgt)
        out.append(((dirty - clean) / inj).item())
    cw.train()
    return float(np.mean(out))

def run_val():
    cw.eval(); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(val_s), a.batch_size):
            zc, ac = batch(val_s[i:i + a.batch_size], train=False)
            tot += cw.forward_ddpm(zc, ac, future_weights=FW).item()
    cw.train()
    return tot / max(1, (len(val_s) + a.batch_size - 1) // a.batch_size)

@torch.no_grad()
def final_frame_mse(n=24):
    """Sampled (DDIM) MSE of the target frame vs the copy-last-history baseline."""
    cw.eval(); m, c = [], []
    for e, t in val_s[:n]:
        H0 = ZR[e][hidx(t)].unsqueeze(0).to(DEV)
        ac = acvec(e, t).unsqueeze(0).to(DEV)
        tgt = ZT[e][fut_idx(t)[-1]].unsqueeze(0).to(DEV)
        m.append(Fn.mse_loss(cw.predict_next_latent(H0, ac, n_steps=10)[:, -1].float(), tgt).item())
        c.append(Fn.mse_loss(H0[:, -1], tgt).item())
    cw.train()
    return float(np.mean(m)), float(np.mean(c))

if RANK == 0 and not a.skip_before:
    print(f"\n[before] val_loss={run_val():.5f}  amplification={amplification():.3f}"
          f"   (<1 = stable loop)", flush=True)
    _m, _c = final_frame_mse()
    print(f"[before] target-frame MSE={_m:.4f}  copy-last baseline={_c:.4f}\n", flush=True)
if WORLD > 1:
    dist.barrier()

# ── train ─────────────────────────────────────────────────────────────────── #
def sync_grads():
    """Average grads across ranks (inf/nan propagates, so every rank skips together)."""
    if WORLD == 1:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float()
                      for p in params])
    dist.all_reduce(flat); flat /= WORLD
    o = 0
    for p in params:
        n = p.numel(); p.grad = flat[o:o + n].view_as(p).to(p.dtype); o += n

for ep in range(1, a.epochs + 1):
    rng.shuffle(train_s); tot, nb = 0.0, 0
    ep_s = train_s[:a.epoch_samples] if a.epoch_samples else train_s
    t0 = time.perf_counter()
    mine = ep_s[RANK::WORLD]
    mine = mine[:len(ep_s) // WORLD]         # equal step counts per rank (collectives)
    for i in range(0, len(mine), a.batch_size):
        zc, ac = batch(mine[i:i + a.batch_size])
        loss = cw.forward_ddpm(zc, ac, future_weights=FW)
        opt.zero_grad()
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
        else:                                   # keep collectives aligned: contribute zero grads
            for q in params: q.grad = torch.zeros_like(q)
        sync_grads()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        if torch.isfinite(loss):
            tot += loss.item(); nb += 1
        if nb and nb % 50 == 0:
            print(f"  ep{ep} {nb}/{(len(mine)+a.batch_size-1)//a.batch_size} "
                  f"loss={tot/nb:.5f}  {(time.perf_counter()-t0)/nb:.2f}s/batch", flush=True)
    if WORLD > 1:
        dist.barrier()
    if RANK == 0:                               # eval + checkpoint on rank 0 only
        amp = amplification()
        print(f"[epoch {ep}] train={tot/max(nb,1):.5f}  val={run_val():.5f}  "
              f"amplification={amp:.3f}", flush=True)
        _m, _c = final_frame_mse()
        print(f"  target-frame MSE={_m:.4f}  copy-last baseline={_c:.4f}", flush=True)
        torch.save({"action_encoder_state": cw.action_encoder.state_dict(),
                    "unet_cross_attn_state": cw.unet_cross_attn_state_dict(),
                    "unet_extra_state": cw.unet_extra_state_dict(),
                    "epoch": ep, "amplification": amp,
                    "meta": {"num_pred": K, "chunk": CHUNK, "pred_stride": PS,
                             "history_stride": M, "num_history": T_H, "views": a.views, "unfreeze": list(a.unfreeze),
                             "pose": POSE, "pose_only": POSE_ONLY, "pose_dim": NP, "pose_scale": a.pose_scale,
                             "pose_mean": (P_MEAN.tolist() if POSE else None),
                             "pose_std": (P_STD.tolist() if POSE else None)}},
                   os.path.join(a.out, f"stage1_rollout_ep{ep}.pt"))
        print(f"  [ckpt] {a.out}/stage1_rollout_ep{ep}.pt", flush=True)
    if WORLD > 1:
        dist.barrier()
print("DONE")
