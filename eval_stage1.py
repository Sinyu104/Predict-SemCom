"""
eval_stage1.py — Downstream action-level evaluation of the Stage 1 predictor.

For each validation clip the predictor is evaluated on single-step prediction:
    z_pred = z_t + predictor(z_t, a_t) / gamma

Three paths are compared:
  real   : z_{t+1}  — actual ViT tokens from the next frame (upper bound)
  pred   : z_pred   — predictor's estimate
  repeat : z_t      — reuse current frame tokens (do-nothing baseline)

Metrics (each path vs. ground-truth VLA action from real tokens):
  action_mse   : ||a_path - a_real||²   mean over all steps

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
    p.add_argument("--use_stub",   action="store_true",
                   help="Use OpenVLAStub (CPU, no model download)")
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
    """Mean per-sample action MSE between two (B, action_dim) tensors."""
    return F.mse_loss(a.float(), b.float()).item()


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
    system.load_state_dict(ckpt["system_state"], strict=False)
    system.eval()
    predictor = system.predictor
    gamma     = CONFIG["gamma_delta"]
    print(f"[eval] Checkpoint : {args.ckpt}  (gamma={gamma})")

    # ── Agent ────────────────────────────────────────────────────────────── #
    agent = build_agent(args.use_stub, CONFIG, unnorm_key_override=args.unnorm_key)

    # ── Validation dataset (same 80/20 split as training) ────────────────── #
    data_path = args.data or CONFIG.get("default_data_path", "data/trajectories.hdf5")
    clip_len  = CONFIG.get("clip_length", 4)

    full_ds = ClipDataset(
        hdf5_path   = data_path,
        obs_height  = CONFIG["obs_height"],
        obs_width   = CONFIG["obs_width"],
        clip_length = clip_len,
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
    print(f"[eval] Val clips : {len(val_ds)}  clip_length={clip_len}")

    instruction = CONFIG["openvla_instruction"]

    # ── Accumulators ─────────────────────────────────────────────────────── #
    mse_pred   = []   # predictor vs real
    mse_repeat = []   # repeat-last-frame vs real
    n_clips    = 0

    n_batches = len(loader) if args.max_clips <= 0 else \
                min(len(loader), math.ceil(args.max_clips / batch_size))
    from tqdm import tqdm
    pbar = tqdm(total=n_batches, desc="Evaluating", unit="batch")

    for frames, actions, _poses in loader:
        # frames  : (B, T, C, H, W)
        # actions : (B, T, action_dim)  a_k at step k
        B, T, C, H, W = frames.shape
        frames  = frames.to(device)
        actions = actions.to(device)

        # Encode all T frames — encode one frame at a time to avoid OOM
        tokens = []
        for t in range(T):
            tok = encode_and_norm(agent, frames[:, t], device)  # (B, N, D)
            tokens.append(tok)
        # tokens: list of T tensors (B, N, D_vit)

        # Single-step prediction for every consecutive pair (t → t+1)
        for t in range(T - 1):
            z_t    = tokens[t]          # (B, N, D_vit) — current frame
            z_real = tokens[t + 1]      # (B, N, D_vit) — next frame (ground truth)
            a_t    = actions[:, t]      # (B, action_dim) — action taken at step t

            # Predictor: outputs γΔ, recover absolute tokens
            gamma_delta = predictor(z_t, a_t)                # (B, N, D_vit)
            z_pred      = z_t + gamma_delta / gamma          # (B, N, D_vit)

            # Actions from the three paths
            a_real   = agent.predict_action_from_tokens(z_real, instruction)   # (B, 7)
            a_pred   = agent.predict_action_from_tokens(z_pred, instruction)   # (B, 7)
            a_repeat = agent.predict_action_from_tokens(z_t,    instruction)   # (B, 7)

            mse_pred.append(action_mse(a_pred,   a_real))
            mse_repeat.append(action_mse(a_repeat, a_real))

        n_clips += B
        pbar.update(1)
        pbar.set_postfix({
            "pred_mse":   f"{mse_pred[-1]:.4f}",
            "repeat_mse": f"{mse_repeat[-1]:.4f}",
            "clips":      n_clips,
        })
        if args.max_clips > 0 and n_clips >= args.max_clips:
            break

    pbar.close()

    # ── Summary ──────────────────────────────────────────────────────────── #
    def mean(lst):
        return sum(lst) / max(len(lst), 1)

    print()
    print("=" * 60)
    print(f"  Stage-1 Predictor Evaluation")
    print(f"  clips evaluated : {n_clips}")
    print(f"  steps evaluated : {len(mse_pred)}")
    print("=" * 60)
    print(f"  Action MSE — predictor   vs real : {mean(mse_pred):.6f}")
    print(f"  Action MSE — repeat-last vs real : {mean(mse_repeat):.6f}")
    print()
    ratio = mean(mse_pred) / max(mean(mse_repeat), 1e-12)
    if ratio < 1.0:
        print(f"  Predictor is {(1 - ratio)*100:.1f}% better than repeat-last-frame.")
    else:
        print(f"  Predictor is {(ratio - 1)*100:.1f}% WORSE than repeat-last-frame.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    evaluate(parse_args())
