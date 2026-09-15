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
a = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T_H  = CONFIG["num_history"]
K    = a.num_pred
M    = a.history_stride
SPAN = (T_H - 1) * M + 1          # frames the strided window reaches back over
CONFIG["num_pred"] = K
os.makedirs(a.out, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────── #
ZT, ZR, A, SNR = [], [], [], []
for f in sorted(glob.glob(a.shards)):
    d = torch.load(f, map_location="cpu")
    ZT.append(d["ZT"]); ZR.append(d["ZR"]); A.append(d["A"])
    SNR.append(d.get("SNR", torch.full((len(d["ZT"]),), float("nan"))))
    print(f"  loaded {f}: {tuple(d['ZT'].shape)}")
ZT = torch.cat(ZT).float(); ZR = torch.cat(ZR).float()
A = torch.cat(A).float();   SNR = torch.cat(SNR)
if a.max_snr is not None:
    keep = SNR <= a.max_snr
    ZT, ZR, A, SNR = ZT[keep], ZR[keep], A[keep], SNR[keep]
n_ep, T = ZT.shape[0], ZT.shape[1]
print(f"[data] {n_ep} episodes x {T} frames  K={K} stride={M} "
      f"(span={(T_H-1)*M}, turnover={(T_H-1)*M/max(K,1):.1f} calls)   "
      f"mean rollout error={((ZR[:, SPAN:]-ZT[:, SPAN:])**2).mean():.4f}")

samples = [(e, t) for e in range(n_ep) for t in range(SPAN, T - K + 1)]
rng = np.random.default_rng(a.seed); rng.shuffle(samples)
n_val = max(1, int(len(samples) * a.val_frac))
val_s, train_s = samples[:n_val], samples[n_val:]
print(f"[data] {len(train_s)} train / {len(val_s)} val samples")

def hidx(t):
    """Strided history indices ending at t-1, oldest first — matches collection."""
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]

def batch(idx):
    """clip = [strided drifted history (T_H) | true targets (K)]."""
    zc = torch.stack([torch.cat([ZR[e][hidx(t)], ZT[e, t:t + K]], 0) for e, t in idx])
    ac = torch.stack([torch.cat([A[e][hidx(t)], A[e, t:t + K]], 0) for e, t in idx])
    return zc.to(DEV), ac.to(DEV)

# ── model ─────────────────────────────────────────────────────────────────── #
cw = CtrlWorldWrapper(
    action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
    svd_path=CONFIG["svd_model_name"], freeze_unet=True,
    finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
    dtype=torch.float16).to(DEV)
ck = torch.load(a.stage1_ckpt, map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
del ck
params = list(cw.trainable_parameters())
opt = torch.optim.AdamW(params, lr=a.lr)
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
        ac = torch.cat([A[e][hidx(t)], A[e, t:t + K]], 0).unsqueeze(0).to(DEV)
        tgt = ZT[e, t].unsqueeze(0).to(DEV)
        clean = Fn.mse_loss(cw.predict_next_latent(H0, ac, n_steps=10)[:, 0].float(), tgt)
        Hb = blur(H0); base = Fn.mse_loss(Hb, H0).item()
        Hc = H0 + (inj / max(base, 1e-12)) ** 0.5 * (Hb - H0)     # same blur all frames
        dirty = Fn.mse_loss(cw.predict_next_latent(Hc, ac, n_steps=10)[:, 0].float(), tgt)
        out.append(((dirty - clean) / inj).item())
    cw.train()
    return float(np.mean(out))

def run_val():
    cw.eval(); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(val_s), a.batch_size):
            zc, ac = batch(val_s[i:i + a.batch_size])
            tot += cw.forward_ddpm(zc, ac).item()
    cw.train()
    return tot / max(1, (len(val_s) + a.batch_size - 1) // a.batch_size)

print(f"\n[before] val_loss={run_val():.5f}  amplification={amplification():.3f}"
      f"   (<1 = stable loop)\n", flush=True)

# ── train ─────────────────────────────────────────────────────────────────── #
for ep in range(1, a.epochs + 1):
    rng.shuffle(train_s); tot, nb = 0.0, 0
    t0 = time.perf_counter()
    for i in range(0, len(train_s), a.batch_size):
        zc, ac = batch(train_s[i:i + a.batch_size])
        loss = cw.forward_ddpm(zc, ac)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        tot += loss.item(); nb += 1
        if nb % 50 == 0:
            print(f"  ep{ep} {nb}/{(len(train_s)+a.batch_size-1)//a.batch_size} "
                  f"loss={tot/nb:.5f}  {(time.perf_counter()-t0)/nb:.2f}s/batch", flush=True)
    amp = amplification()
    print(f"[epoch {ep}] train={tot/max(nb,1):.5f}  val={run_val():.5f}  "
          f"amplification={amp:.3f}", flush=True)
    torch.save({"action_encoder_state": cw.action_encoder.state_dict(),
                "unet_cross_attn_state": cw.unet_cross_attn_state_dict(),
                "epoch": ep, "amplification": amp},
               os.path.join(a.out, f"stage1_rollout_ep{ep}.pt"))
    print(f"  [ckpt] {a.out}/stage1_rollout_ep{ep}.pt", flush=True)
print("DONE")
