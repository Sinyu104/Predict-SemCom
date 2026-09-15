"""Collect CLOSED-LOOP rollout data for Stage-1 fine-tuning.

Stage 1 is trained on ground-truth history with i.i.d. Gaussian augmentation.  At
deployment the history is the receiver's own reconstruction, whose error is
neither i.i.d. nor Gaussian: it is spatially smooth AND temporally correlated —
the six history frames drift together and eventually collapse to near-copies of
one another, which reads as "nothing is moving".  Measured amplification of an
i.i.d. blur perturbation is ~0.5, but the closed loop shows >2, so the temporal
structure is the part that matters and no per-frame augmentation reproduces it.

This script runs the real receiver loop over recorded demos and saves the
histories it actually induces, so Stage 1 can be fine-tuned on them:

    for each episode:
        history <- 6 clean latents
        repeat:
            z_hat   = CtrlWorld(history, actions)        # drifts
            s~      = channel(JSCC(z_t))                 # from the TRUE z_t
            z~      = Refinement(z_hat, s~)
            history <- history + [z~]                    # the feedback that drifts
            record (history, actions, TRUE z_t)

No simulator and no policy are involved — the loop is entirely receiver-side.

Output shard: {"ZT": true latents, "ZR": rollout reconstructions, "A": actions},
each (n_episodes, T, ...).  Training builds a sample as
history = ZR[t-num_history:t], action = A[t-num_history:t+1], target = ZT[t].

Run one process per GPU:
    python experiments/rollout_collect.py --shard 0 --nshards 4 \
        --out outputs/rollout/shard0.pt --ddim_steps 3 --rollout_len 24
"""
import argparse, glob, os, pathlib, sys, time

import h5py
import numpy as np
import torch

REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from config import CONFIG                       # noqa: E402
from models import SemComSystem                 # noqa: E402
from vae_wrapper import VAEWrapper              # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", type=str, required=True)
ap.add_argument("--episodes_per_task", type=int, default=40)
ap.add_argument("--rollout_len", type=int, default=24,
                help="Number of FRAMES rolled out per episode (advances num_pred at a time).")
ap.add_argument("--num_pred", type=int, default=1,
                help="K: frames predicted per model call. Deployment compounding depth is "
                     "rollout_len/K, so K=8 gives 8x fewer autoregressive steps than K=1. "
                     "Must match the K the Stage-1 checkpoint was trained at.")
ap.add_argument("--ddim_steps", type=int, default=3,
                help="Ctrl-World DDIM steps during collection. Dominates runtime; "
                     "check quality against the deployment setting before lowering.")
ap.add_argument("--stage1_ckpt", type=str,
                default="outputs/stage1_5cube_cam1_K8/stage1_best.pt")
ap.add_argument("--stage2_ckpt", type=str,
                default="outputs/stage2_twophase_full/phase2.pt")
ap.add_argument("--channel", type=str, default=None,
                choices=["awgn", "rayleigh", "cdl"])
ap.add_argument("--snr_db", type=float, default=None)
ap.add_argument("--snr_list", nargs="+", type=float, default=None,
                help="Sample an SNR per episode from this list instead of a fixed "
                     "--snr_db, so the drift severity is mixed. Stage 1 trained on "
                     "one SNR is tuned to one drift level and will not transfer "
                     "across an SNR sweep.")
ap.add_argument("--channel_list", nargs="+", default=None,
                choices=["awgn", "rayleigh", "cdl"],
                help="Sample a channel type per episode. CDL carries a ~5 dB ZF "
                     "penalty vs AWGN, so its drift is harsher at equal nominal SNR.")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--history_stride", type=int, default=1,
                help="m: spacing between history frames. With m=1 (consecutive) and "
                     "K=8, one call advances past the entire 6-frame history, so the "
                     "context becomes 100%% synthetic after a SINGLE call. Ctrl-World "
                     "spaces history 1-2 s apart so only ~1 of 7 frames turns over per "
                     "call. History span is (num_history-1)*m; aim for span/K ~ 7.")
ap.add_argument("--reanchor", type=int, default=0,
                help="Every N frames, replace the history entry with the TRUE latent — "
                     "a keyframe, as H.265 does with I-frames. Bounds drift to N frames "
                     "instead of the whole episode. 0 = never re-anchor. Costs rate: one "
                     "full-rate frame per N, which must be included in any rate comparison.")
ap.add_argument("--pure_wm", action="store_true",
                help="Feed z_hat straight back as history, with no JSCC/channel/refinement. "
                     "Isolates the world model's own autoregressive stability and removes "
                     "any dependence on the current Stage 2, so the data does not go stale "
                     "when Stage 2 changes. It is also the HARDER case: z_hat drifts worse "
                     "than the refined z~ does, so a model trained here handles refined "
                     "history at least as well.")
ap.add_argument("--demos_file", type=str, default="demos.hdf5",
                help="demos.hdf5 (clean) or demos_disturbed.hdf5. Stage-2 MUST use "
                     "disturbed: with no disturbance z_hat ~= z_t, there is nothing "
                     "for the channel to correct, and the decoder learns to hold "
                     "z_hat and ignore s~.")
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = f"cuda:{a.shard}" if torch.cuda.device_count() > a.shard else "cuda:0"
T_H  = CONFIG["num_history"]
K    = a.num_pred
M    = a.history_stride
SEED = (T_H - 1) * M + 1        # clean frames needed before the first strided window
CONFIG["num_pred"] = K
CONFIG["clip_length"] = T_H + K
if a.channel: CONFIG["channel_type"] = a.channel
if a.snr_db is not None: CONFIG["snr_db"] = a.snr_db

# ── episode index, sharded ────────────────────────────────────────────────── #
index = []
for t in a.tasks:
    p = os.path.join(REPO, "data", t, a.demos_file)
    if not os.path.isfile(p):
        print(f"[shard {a.shard}] skip {t}: no {a.demos_file}", flush=True)
        continue
    with h5py.File(p, "r") as f:
        eps = sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit())))
        for e in eps[:a.episodes_per_task]:
            if len(f[e]["observations_cam1"]) >= SEED + a.rollout_len + K:
                index.append((p, e))
index = index[a.shard::a.nshards]
print(f"[shard {a.shard}] {len(index)} episodes from {a.demos_file}  K={K} "
      f"stride={M} seed={SEED} (span={(T_H-1)*M}, turnover={(T_H-1)*M/max(K,1):.1f} calls)  "
      f"ddim={a.ddim_steps}  pure_wm={a.pure_wm}  reanchor={a.reanchor}", flush=True)

# ── models ────────────────────────────────────────────────────────────────── #
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV)
cw = CtrlWorldWrapper(
    action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
    svd_path=CONFIG["svd_model_name"], freeze_unet=True,
    finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
    dtype=torch.float16).to(DEV)
ck = torch.load(a.stage1_ckpt, map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
del ck
cw.eval()

sysm = None
if not a.pure_wm:
    sysm = SemComSystem(CONFIG).to(DEV)
    sk = torch.load(a.stage2_ckpt, map_location="cpu")
    sysm.load_state_dict(sk.get("system_state", sk), strict=False)
    del sk
    sysm.eval()

T_SECOND = CONFIG.get("noise_level_second", 0)
ZT_all, ZR_all, A_all, ZH_all = [], [], [], []
SNR_all, CH_all = [], []
rng = np.random.default_rng(a.seed + a.shard)
CH_CODES = {"awgn": 0, "rayleigh": 1, "cdl": 2}
t_start = time.perf_counter()

for i, (p, e) in enumerate(index):
    # Per-episode channel condition, so one shard spans the whole drift range.
    ep_snr = float(rng.choice(a.snr_list)) if a.snr_list else CONFIG["snr_db"]
    ep_ch  = str(rng.choice(a.channel_list)) if a.channel_list else CONFIG["channel_type"]
    if sysm is not None:
        sysm.channel.snr_db       = ep_snr
        sysm.channel.channel_type = ep_ch
    with h5py.File(p, "r") as f:
        g = f[e]
        n = SEED + a.rollout_len
        x = torch.from_numpy(np.array(g["observations_cam1"][:n])).float().div_(255.)
        x = x.permute(0, 3, 1, 2)
        # +K so the final chunk's action window is still full length
        act = torch.from_numpy(np.array(g["actions"][:n + K], dtype=np.float32))

    with torch.no_grad():
        Z = torch.cat([vae.encode(x[j:j + 4].to(DEV)) for j in range(0, len(x), 4)])
        hist = [Z[k] for k in range(SEED)]       # clean seed, long enough for the stride
        recon = [Z[k] for k in range(SEED)]      # ZR mirrors ZT over the seed
        zhats = []                               # z_hat per rollout step, for Stage 2

        # Advance K frames per call: one prediction, then refine each of the K
        # frames with its own s~.  Compounding happens only BETWEEN calls, which
        # is why larger K means a shallower autoregressive chain.
        for t in range(SEED, n, K):
            kk = min(K, n - t)
            # strided window: frames at t-1, t-1-M, t-1-2M, ... oldest first
            hidx = [max(0, len(hist) - 1 - i * M) for i in range(T_H)][::-1]
            H = torch.stack([hist[j] for j in hidx], 0).unsqueeze(0)
            a_win = torch.cat([act[torch.tensor(hidx)], act[t:t + K]], 0
                              ).unsqueeze(0).to(DEV)             # (1, T_H+K, act_dim)
            z_pred = cw.predict_next_latent(H, a_win, n_steps=a.ddim_steps).float()
            for i in range(kk):
                z_hat = z_pred[:, i]
                if a.reanchor > 0 and (t + i) % a.reanchor == 0:
                    z_til = Z[t + i].unsqueeze(0)      # keyframe: receiver gets the truth
                elif a.pure_wm:
                    z_til = z_hat                      # world model feeds itself
                else:
                    _, _, s = sysm.jscc_encoder(Z[t + i].unsqueeze(0), sample=False)
                    s_tilde = sysm.channel(s)
                    z_til = sysm.refinement_diffusion.sdedit_refine(
                        z_hat, s_tilde, noise_level=T_SECOND, n_steps=a.ddim_steps)
                hist.append(z_til[0])
                recon.append(z_til[0])
                zhats.append(z_hat[0])

    ZT_all.append(Z.half().cpu())
    ZH_all.append(torch.stack(zhats, 0).half().cpu())   # (rollout_len, C, H, W)
    ZR_all.append(torch.stack(recon, 0).half().cpu())
    A_all.append(act[:n].cpu())
    SNR_all.append(ep_snr); CH_all.append(CH_CODES[ep_ch])

    if i % 5 == 0:
        el = time.perf_counter() - t_start
        print(f"[shard {a.shard}] {i+1}/{len(index)}  {el/max(i+1,1):.1f}s/ep  "
              f"eta {el/max(i+1,1)*(len(index)-i-1)/60:.0f} min", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
ZT = torch.stack(ZT_all); ZR = torch.stack(ZR_all); A = torch.stack(A_all)
SNR = torch.tensor(SNR_all); CH = torch.tensor(CH_all)
ZH = torch.stack(ZH_all)                      # (n_ep, rollout_len, C, H, W)
# Stage-2 consumes (ZT_roll, ZH) pairs: z_hat as produced by the CLOSED LOOP,
# aligned with the true latent at the same step.  twophase.py reshapes both to
# (-1, C, H, W), so the leading dims only have to match.
ZT_roll = ZT[:, SEED:]
torch.save({"ZT": ZT, "ZR": ZR, "A": A, "SNR": SNR, "CH": CH,
            "ZH": ZH, "ZT_roll": ZT_roll,
            "meta": {"ddim_steps": a.ddim_steps, "rollout_len": a.rollout_len,
                     "num_history": T_H, "num_pred": K, "history_stride": M,
                     "seed": SEED, "ch_codes": CH_CODES,
                     "snr_list": a.snr_list, "channel_list": a.channel_list,
                     "stage1": a.stage1_ckpt, "stage2": a.stage2_ckpt,
                     "demos_file": a.demos_file}}, a.out)

err = ((ZR[:, T_H:].float() - ZT[:, T_H:].float()) ** 2).mean(dim=(1, 2, 3, 4))
print(f"[shard {a.shard}] saved {a.out}  ZT={tuple(ZT.shape)}  "
      f"mean rollout error={err.mean():.4f}", flush=True)
for s_ in sorted(set(SNR_all)):
    msk = SNR == s_
    print(f"    snr={s_:5.1f} dB  n={int(msk.sum()):3d}  "
          f"rollout error={err[msk].mean():.4f}", flush=True)
