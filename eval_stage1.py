"""
eval_stage1.py — Downstream action-level evaluation of the Stage 1 predictor.

For each validation clip the predictor is evaluated at rollout horizons 1–(T-1):
  Horizon h: start from frame t, autoregressively predict h steps ahead.
             All valid starting frames t (where t+h < T) are used.

  z_pred_h = rollout(z_t, actions[t:t+h], h steps)

Three paths are compared at each horizon:
  real   : z_{t+h}  — actual ViT tokens (upper bound)
  pred   : z_pred_h — predictor rollout
  repeat : z_t      — reuse starting frame (do-nothing baseline)

Metrics per horizon:
  Latent  : L1, MSE, L2 (RMSE), cosine similarity  (z_pred vs z_real)
  Action  : MSE vs ground-truth dataset action  (a_pred, a_real_tok, a_repeat)
  Action  : MSE vs VLA-from-real-tokens         (a_pred, a_repeat)

Run (real VLA, fine-tuned):
  python eval_stage1.py \
      --ckpt outputs/stage1_best.pt \
      --data data/stage12_clean.hdf5

Run (stub, for shape checking without GPU / model download):
  python eval_stage1.py \
      --ckpt outputs/stage1_best.pt \
      --data data/stage12_clean.hdf5 \
      --use_stub
"""

import argparse
import math
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

from config  import CONFIG
from models  import SemComSystem
from dataset import ClipDataset


# ─────────────────────────────────────────────────────────────────────────── #
#  CLI                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 downstream evaluation")
    p.add_argument("--ckpt",       type=str, default="outputs/stage1_best.pt")
    p.add_argument("--data",       type=str, default=None,
                   help="Path to HDF5 (default: config default_data_path)")
    p.add_argument("--max_clips",  type=int, default=200,
                   help="Max validation clips to evaluate (0 = all)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--unnorm_key", type=str, default=None,
                   help="OpenVLA unnorm_key override (e.g. bridge_orig). "
                        "Use this if your unnorm_key is not in the loaded model's norm_stats.")
    p.add_argument("--use_stub",     action="store_true",
                   help="Use OpenVLAStub (CPU, no model download)")
    p.add_argument("--latent_only", action="store_true",
                   help="Skip VLA action decoding; report only latent-space metrics (fast)")
    p.add_argument("--clip_stride", type=int, default=None,
                   help="Clip stride (default: read from config clip_stride, fallback 1)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────── #
#  Agent loader (mirrors main.py build_agent)                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def build_agent(use_stub: bool, config: dict, unnorm_key_override: str | None = None):
    unnorm_key = unnorm_key_override or config.get("openvla_unnorm_key", "bridge_orig")

    if use_stub or not torch.cuda.is_available():
        from openvla_agent import OpenVLAStub
        print("[eval] Using OpenVLAStub")
        return OpenVLAStub(
            N_patches   = config["N_patches"],
            D_vit       = config["D_vit"],
            action_dim  = config["action_dim"],
            instruction = config["openvla_instruction"],
        )

    from openvla_agent import OpenVLAAgent
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
    try:
        from vla_server import _resolve_model_dir
        ft_dir     = config.get("openvla_finetune_dir", "")
        model_name = _resolve_model_dir(ft_dir) \
                     if (ft_dir and os.path.isdir(ft_dir)) \
                     else config["openvla_model_name"]
    except ImportError:
        model_name = config["openvla_model_name"]

    print(f"[eval] Loading OpenVLA : {model_name}")
    print(f"[eval] unnorm_key      : {unnorm_key}")
    return OpenVLAAgent(
        instruction = config["openvla_instruction"],
        unnorm_key  = unnorm_key,
        model_name  = model_name,
        device      = config.get("openvla_device_map", "auto"),
        quantize    = config.get("openvla_quantize", False),
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def encode_and_norm(agent, obs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Encode (B, C, H, W) → (B, N, D_vit) with layer-norm (mirrors trainer)."""
    tokens = agent.encode_image(obs).to(device)
    return F.layer_norm(tokens, [tokens.shape[-1]]).detach()


def action_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a.float().cpu(), b.float().cpu()).item()


def latent_metrics(z_pred: torch.Tensor, z_real: torch.Tensor) -> dict:
    """L1, MSE, L2 (RMSE), cosine-sim between (B, N, D) latent tensors.
    Cosine similarity is computed per patch (dim=-1, across D_vit) to match
    the trainer's val_cos metric."""
    zp   = z_pred.float()
    zr   = z_real.float()
    diff = zp - zr
    mse  = diff.pow(2).mean().item()
    return {
        "l1":  diff.abs().mean().item(),
        "mse": mse,
        "l2":  math.sqrt(mse),
        "cos": F.cosine_similarity(zp, zr, dim=-1).mean().item(),
    }


def wyner_ziv_kl(mse_pred: float, mse_start: float, mse_pred_vs_start: float) -> dict:
    """
    Gaussian Wyner-Ziv KL metrics (per latent element).

    Models:
      p(z_real | z_pred)  = N(z_pred,  σ_pred²  · I),  σ_pred²  = mse_pred
      p(z_real | z_start) = N(z_start, σ_start² · I),  σ_start² = mse_start

    KL(p(z_real|z_pred) ‖ p(z_real|z_start)) per element:
      = 0.5 * (σ_pred²/σ_start² − 1 − ln(σ_pred²/σ_start²))
        + mse_pred_vs_start / (2 · σ_start²)

    bits_saved/dim = 0.5 · log2(σ_start² / σ_pred²)
      Positive → predictor reduces the bits needed to describe z_real.
    """
    eps = 1e-12
    ratio = mse_pred / max(mse_start, eps)
    kl_per_dim = 0.5 * (ratio - 1.0 - math.log(ratio + eps)) \
                 + mse_pred_vs_start / (2.0 * max(mse_start, eps))
    bits_saved = 0.5 * math.log2(max(mse_start, eps) / max(mse_pred, eps))
    return {"kl_per_dim": kl_per_dim, "bits_saved_per_dim": bits_saved}


def predict_h1_full_clip(predictor, tokens_stacked: torch.Tensor,
                         actions: torch.Tensor, gamma: float) -> list:
    """
    Teacher-forced horizon-1 predictions using full-clip forward_clip —
    exactly matches the trainer's val_cos computation.

    tokens_stacked : (B, T, N, D_vit)
    actions        : (B, T-1, action_dim)
    Returns list of T-1 tensors (B, N, D_vit), one per step.
    """
    preds_tf = predictor.forward_clip(tokens_stacked, actions)  # (B, T-1, N, D)
    T = tokens_stacked.shape[1]
    return [tokens_stacked[:, t] + preds_tf[:, t] / gamma
            for t in range(T - 1)]


def rollout_ar(predictor, tokens_stacked: torch.Tensor, actions: torch.Tensor,
               start: int, n_steps: int, gamma: float) -> torch.Tensor:
    """
    Autoregressive rollout for n_steps starting from frame `start`,
    using forward_clip with accumulated predicted context (mirrors
    rollout_2step / rollout_3step in models.py).

    tokens_stacked : (B, T, N, D_vit)  — real tokens (only tokens[:, start]
                     is used as the seed; subsequent frames come from predictions)
    actions        : (B, T-1, action_dim)
    Returns z_hat after n_steps  : (B, N, D_vit)
    """
    seq = [tokens_stacked[:, start]]          # seed: real frame at `start`
    for k in range(n_steps):
        dummy      = torch.zeros_like(seq[0])
        tokens_seq = torch.stack(seq + [dummy], dim=1)        # (B, k+2, N, D)
        actions_k  = actions[:, start:start + k + 1]          # (B, k+1, action_dim)
        delta      = predictor.forward_clip(tokens_seq, actions_k)[:, k]
        seq.append(seq[-1] + delta / gamma)
    return seq[-1]


# ─────────────────────────────────────────────────────────────────────────── #
#  Evaluation                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device : {device}")

    # ── Checkpoint ───────────────────────────────────────────────────────── #
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    system = SemComSystem(CONFIG).to(device)
    # Strip DDP '.module.' prefix inserted when saved from multi-GPU training
    state = {k.replace(".module.", "."): v
             for k, v in ckpt["system_state"].items()}
    missing, unexpected = system.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[eval] WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    system.eval()
    predictor = system.predictor
    gamma     = CONFIG.get("gamma_delta", 50.0)
    print(f"[eval] Checkpoint : {args.ckpt}  gamma={gamma}")

    # ── Agent ────────────────────────────────────────────────────────────── #
    agent = build_agent(args.use_stub, CONFIG, unnorm_key_override=args.unnorm_key)
    latent_only = args.latent_only
    if latent_only:
        print("[eval] --latent_only: skipping VLA action decoding")

    # ── Validation dataset (same 80/20 split as training) ────────────────── #
    data_path = args.data or CONFIG.get("default_data_path", "data/trajectories.hdf5")
    clip_len    = CONFIG.get("clip_length", 4)
    clip_stride = args.clip_stride if args.clip_stride is not None \
                  else CONFIG.get("clip_stride", 1)

    full_ds = ClipDataset(
        hdf5_path    = data_path,
        obs_height   = CONFIG["obs_height"],
        obs_width    = CONFIG["obs_width"],
        obs_channels = CONFIG["obs_channels"],
        clip_length  = clip_len,
        stride       = clip_stride,
    )
    n_val   = max(1, int(0.2 * len(full_ds)))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG.get("seed", 42)),
    )
    batch_size = args.batch_size
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[eval] Val clips : {len(val_ds)}  clip_length={clip_len}  stride={clip_stride}")

    instruction = CONFIG["openvla_instruction"]
    max_horizon = clip_len - 1  # e.g. 3 for T=4

    # ── Accumulators keyed by horizon ────────────────────────────────────── #
    # Each key h ∈ {1, …, max_horizon} maps to a list of scalar values.
    acc = {h: defaultdict(list) for h in range(1, max_horizon + 1)}
    # Per-position clip accumulator: key t ∈ {0, …, max_horizon-1}
    # t=0 is cold-start (only z_0 as context); t>0 has t prior frames.
    acc_clip = {t: defaultdict(list) for t in range(max_horizon)}
    n_clips = 0

    n_batches = len(loader) if args.max_clips <= 0 else \
                min(len(loader), math.ceil(args.max_clips / batch_size))
    from tqdm import tqdm
    pbar = tqdm(total=n_batches, desc="Evaluating", unit="batch")

    for frames, actions, _poses in loader:
        # frames  : (B, T, C, H, W)
        # actions : (B, T-1, action_dim)  a_k at step k
        B, T, C, H, W = frames.shape
        frames  = frames.to(device)
        actions = actions.to(device)

        # Encode all T frames
        tokens = []
        for t in range(T):
            tokens.append(encode_and_norm(agent, frames[:, t], device))
        # tokens: list of T tensors each (B, N, D_vit)
        tokens_stacked = torch.stack(tokens, dim=1)   # (B, T, N, D_vit)

        # Horizon 1 — teacher-forced full-clip forward_clip (matches trainer val_cos)
        z_preds_h1 = predict_h1_full_clip(predictor, tokens_stacked, actions, gamma)

        # Per-position clip metrics: break out each step t→t+1 separately.
        # z_preds_h1[t] predicts z_{t+1} with context z_0...z_t via block-causal
        # attention, so t=0 is cold-start and t>0 has growing temporal history.
        for t in range(T - 1):
            lm = latent_metrics(z_preds_h1[t], tokens[t + 1])
            acc_clip[t]["l1"].append(lm["l1"])
            acc_clip[t]["mse"].append(lm["mse"])
            acc_clip[t]["l2"].append(lm["l2"])
            acc_clip[t]["cos"].append(lm["cos"])

        # Pre-compute VLA actions for every real token once (cache across horizons).
        if not latent_only:
            a_tok = [agent.predict_action_from_tokens(tokens[t], instruction)
                     for t in range(T)]

        # Evaluate every valid (start, horizon) pair
        for h in range(1, max_horizon + 1):
            for t in range(T - h):
                z_start = tokens[t]          # (B, N, D_vit)
                z_real  = tokens[t + h]      # (B, N, D_vit)  ground-truth target
                # action at the destination frame; None when t+h is the last frame
                # (actions has T-1 entries so index T-1 is out of bounds)
                a_gt = actions[:, t + h] if (t + h) < actions.shape[1] else None

                # Predict: h=1 uses teacher-forced full-clip; h>1 uses AR rollout
                if h == 1:
                    z_pred = z_preds_h1[t]
                else:
                    z_pred = rollout_ar(predictor, tokens_stacked, actions, t, h, gamma)

                # Latent metrics (fast — pure tensor ops)
                lm = latent_metrics(z_pred, z_real)
                acc[h]["lat_l1"].append(lm["l1"])
                acc[h]["lat_mse"].append(lm["mse"])
                acc[h]["lat_l2"].append(lm["l2"])
                acc[h]["lat_cos"].append(lm["cos"])

                # z_pred vs z_start: how much the predictor moved the tokens
                lm_vs_input = latent_metrics(z_pred, z_start)
                acc[h]["inp_l1"].append(lm_vs_input["l1"])
                acc[h]["inp_mse"].append(lm_vs_input["mse"])
                acc[h]["inp_l2"].append(lm_vs_input["l2"])
                acc[h]["inp_cos"].append(lm_vs_input["cos"])

                # z_start vs z_real: repeat/no-prediction baseline
                lm_rep = latent_metrics(z_start, z_real)
                acc[h]["rep_l1"].append(lm_rep["l1"])
                acc[h]["rep_mse"].append(lm_rep["mse"])
                acc[h]["rep_cos"].append(lm_rep["cos"])

                # Gaussian Wyner-Ziv KL: predictor vs repeat as side information
                wz = wyner_ziv_kl(lm["mse"], lm_rep["mse"], lm_vs_input["mse"])
                acc[h]["kl_per_dim"].append(wz["kl_per_dim"])
                acc[h]["bits_saved"].append(wz["bits_saved_per_dim"])

                if not latent_only:
                    # Reuse cached real-token actions; only decode the predicted tokens
                    a_real_tok = a_tok[t + h]
                    a_repeat   = a_tok[t]
                    a_pred_tok = agent.predict_action_from_tokens(z_pred, instruction)

                    # vs VLA-from-real-tokens
                    acc[h]["act_pred_vs_real"].append(action_mse(a_pred_tok, a_real_tok))
                    acc[h]["act_rep_vs_real"].append(action_mse(a_repeat,    a_real_tok))

                    # vs ground-truth dataset action (skipped at last frame boundary)
                    if a_gt is not None:
                        acc[h]["act_pred_vs_gt"].append(action_mse(a_pred_tok, a_gt))
                        acc[h]["act_real_vs_gt"].append(action_mse(a_real_tok, a_gt))
                        acc[h]["act_rep_vs_gt"].append(action_mse(a_repeat,    a_gt))

        n_clips += B
        pbar.update(1)
        h1 = acc[1]
        postfix = {"h1_cos": f"{h1['lat_cos'][-1]:.4f}" if h1["lat_cos"] else "n/a",
                   "clips": n_clips}
        if not latent_only and h1["act_pred_vs_real"]:
            postfix["h1_act_mse"] = f"{h1['act_pred_vs_real'][-1]:.4f}"
        pbar.set_postfix(postfix)
        if args.max_clips > 0 and n_clips >= args.max_clips:
            break

    pbar.close()

    # ── Summary ──────────────────────────────────────────────────────────── #
    def mean(lst):
        return sum(lst) / max(len(lst), 1)

    W = 68
    print()
    print("=" * W)
    print(f"  Stage-1 Predictor Evaluation — {args.ckpt}")
    print(f"  clips evaluated : {n_clips}   clip_length={clip_len}   stride={clip_stride}   gamma={gamma}")
    print("=" * W)

    for h in range(1, max_horizon + 1):
        a = acc[h]
        n = len(a["lat_mse"])
        print(f"\n  ── Horizon {h} ({n} samples) {'─'*(W-22)}")
        print(f"  Latent (z_pred vs z_real):")
        print(f"    L1  (MAE)         : {mean(a['lat_l1']):.6f}")
        print(f"    MSE               : {mean(a['lat_mse']):.6f}")
        print(f"    L2  (RMSE)        : {mean(a['lat_l2']):.6f}")
        print(f"    Cosine similarity : {mean(a['lat_cos']):.6f}")
        print(f"  Latent (z_start vs z_real — repeat/no-pred baseline):")
        print(f"    L1  (MAE)         : {mean(a['rep_l1']):.6f}")
        print(f"    MSE               : {mean(a['rep_mse']):.6f}")
        print(f"    Cosine similarity : {mean(a['rep_cos']):.6f}")
        print(f"  Latent (z_pred vs z_start — how much predictor moved):")
        print(f"    L1  (MAE)         : {mean(a['inp_l1']):.6f}")
        print(f"    MSE               : {mean(a['inp_mse']):.6f}")
        print(f"    L2  (RMSE)        : {mean(a['inp_l2']):.6f}")
        print(f"    Cosine similarity : {mean(a['inp_cos']):.6f}")
        print(f"  Wyner-Ziv (Gaussian, pred vs repeat as side info):")
        print(f"    KL per dim        : {mean(a['kl_per_dim']):.6f}  nats")
        bits = mean(a['bits_saved'])
        tag  = "predictor saves" if bits >= 0 else "predictor COSTS"
        print(f"    Bits saved/dim    : {bits:.6f}  ({tag} {abs(bits):.4f} bits/dim vs repeat)")
        if not latent_only:
            print(f"  Action MSE vs VLA-from-real-tokens:")
            print(f"    predictor         : {mean(a['act_pred_vs_real']):.6f}")
            print(f"    repeat-start-frame: {mean(a['act_rep_vs_real']):.6f}")
            print(f"  Action MSE vs ground-truth (dataset) action:")
            print(f"    predictor         : {mean(a['act_pred_vs_gt']):.6f}")
            print(f"    VLA-from-real-tok : {mean(a['act_real_vs_gt']):.6f}")
            print(f"    repeat-start-frame: {mean(a['act_rep_vs_gt']):.6f}")
            ratio = mean(a["act_pred_vs_real"]) / max(mean(a["act_rep_vs_real"]), 1e-12)
            tag = f"{(1-ratio)*100:.1f}% better" if ratio < 1 else f"{(ratio-1)*100:.1f}% WORSE"
            print(f"  → Predictor is {tag} than repeat (vs real-tok)")
            ratio_gt = mean(a["act_pred_vs_gt"]) / max(mean(a["act_rep_vs_gt"]), 1e-12)
            tag_gt = f"{(1-ratio_gt)*100:.1f}% better" if ratio_gt < 1 else f"{(ratio_gt-1)*100:.1f}% WORSE"
            print(f"  → Predictor is {tag_gt} than repeat (vs GT)")

    # ── Per-position clip prediction summary ─────────────────────────────── #
    print()
    print("=" * W)
    print("  Per-position clip prediction (1-step teacher-forced, full prior context)")
    print("  Each row t→t+1 predicts z_{t+1} using real z_0...z_t via block-causal attn.")
    print("  Shows whether temporal context improves 1-step prediction quality.")
    print(f"  {'Pos':>4}  {'Context':>18}  {'MSE':>10}  {'L2(RMSE)':>10}  {'Cosine':>8}  {'L1':>10}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")
    for t in range(max_horizon):
        a    = acc_clip[t]
        ctx  = "cold start" if t == 0 else f"{t} prior frame{'s' if t > 1 else ''}"
        n    = len(a["mse"])
        print(f"  {t}→{t+1}  {ctx:>18}  "
              f"{mean(a['mse']):>10.6f}  "
              f"{mean(a['l2']):>10.6f}  "
              f"{mean(a['cos']):>8.6f}  "
              f"{mean(a['l1']):>10.6f}  "
              f"({n} samples)")
    print()
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    evaluate(parse_args())
