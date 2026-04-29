"""
check_stride.py — Inter-frame MSE and cosine similarity at different strides,
measured in ViT latent space (after visual encoder + layer norm) and optionally
in compact space (after TokenEncoder).

Run:
    python check_stride.py \
        --data data/stage12_clean.hdf5 \
        --ckpt outputs/stage1_best.pt \
        --n_pairs 200
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import h5py

from config  import CONFIG
from models  import SemComSystem
from dataset import ClipDataset
from torch.utils.data import DataLoader, random_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    type=str, default="data/stage12_clean.hdf5")
    p.add_argument("--ckpt",    type=str, default="outputs/stage1_best.pt")
    p.add_argument("--n_pairs", type=int, default=200,
                   help="Number of frame pairs to sample per stride")
    p.add_argument("--max_stride", type=int, default=4)
    p.add_argument("--use_stub",   action="store_true")
    return p.parse_args()


def build_agent(use_stub, config):
    if use_stub or not torch.cuda.is_available():
        from openvla_agent import OpenVLAStub
        print("[check_stride] Using OpenVLAStub")
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
        ft_dir = config.get("openvla_finetune_dir", "")
        model_name = _resolve_model_dir(ft_dir) \
                     if (ft_dir and os.path.isdir(ft_dir)) \
                     else config["openvla_model_name"]
    except ImportError:
        model_name = config["openvla_model_name"]
    print(f"[check_stride] Loading OpenVLA: {model_name}")
    return OpenVLAAgent(
        instruction = config["openvla_instruction"],
        unnorm_key  = config.get("openvla_unnorm_key", "franka_isaac"),
        model_name  = model_name,
        device      = config.get("openvla_device_map", "auto"),
        quantize    = config.get("openvla_quantize", False),
    )


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[check_stride] Device: {device}")

    # ── Load agent (ViT encoder) ─────────────────────────────────────── #
    agent = build_agent(args.use_stub, CONFIG)

    # ── Load TokenEncoder from checkpoint ────────────────────────────── #
    system = SemComSystem(CONFIG).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        system.load_state_dict(ckpt["system_state"], strict=False)
        print(f"[check_stride] Loaded checkpoint: {args.ckpt}")
    else:
        print(f"[check_stride] No checkpoint found at {args.ckpt}, using random TokenEncoder")
    system.eval()
    tok_enc = system.token_encoder

    # ── Build dataset with large enough clip_length ───────────────────── #
    max_stride   = args.max_stride
    clip_length  = max_stride + 1   # need t and t+max_stride in the same clip

    full_ds = ClipDataset(
        hdf5_path   = args.data,
        obs_height  = CONFIG["obs_height"],
        obs_width   = CONFIG["obs_width"],
        clip_length = clip_length,
    )
    n_val   = max(1, int(0.2 * len(full_ds)))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG.get("seed", 42)),
    )
    loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=0)
    print(f"[check_stride] Val clips: {len(val_ds)}  clip_length={clip_length}")
    print()

    # ── Collect frame pairs ───────────────────────────────────────────── #
    # For each stride, store (z_t, z_{t+stride}) pairs in latent space
    # key: stride → list of (z_t, z_next) tuples, each (N, D_vit)
    vit_pairs   = {s: [] for s in range(1, max_stride + 1)}   # ViT space
    comp_pairs  = {s: [] for s in range(1, max_stride + 1)}   # compact space

    n_collected = 0
    for frames, actions, _poses in loader:
        if n_collected >= args.n_pairs:
            break

        B, T, C, H, W = frames.shape
        frames = frames.to(device)

        # Encode all T frames through ViT
        z_list = []
        for t in range(T):
            raw   = agent.encode_image(frames[:, t]).to(device)
            z_ln  = F.layer_norm(raw, [raw.shape[-1]])   # (B, N, D_vit)
            z_list.append(z_ln.detach().cpu())

        # Encode all T frames through TokenEncoder
        c_list = []
        for t in range(T):
            c = tok_enc(z_list[t].to(device)).detach().cpu()   # (B, N, D_compact)
            c_list.append(c)

        # Collect pairs for each stride
        for stride in range(1, max_stride + 1):
            if stride >= T:
                continue
            # Use t=0 as the anchor frame
            z_t    = z_list[0]    # (B, N, D_vit)
            z_next = z_list[stride]
            c_t    = c_list[0]
            c_next = c_list[stride]
            for b in range(B):
                vit_pairs[stride].append((z_t[b], z_next[b]))
                comp_pairs[stride].append((c_t[b], c_next[b]))

        n_collected += B

    # ── Compute metrics ───────────────────────────────────────────────── #
    def compute_stats(pairs):
        mse_vals = []
        cos_vals = []
        for a, b in pairs[:args.n_pairs]:
            a = a.float()
            b = b.float()
            mse_vals.append(F.mse_loss(a, b).item())
            cos_vals.append(F.cosine_similarity(
                a.reshape(1, -1), b.reshape(1, -1)
            ).item())
        return float(np.mean(mse_vals)), float(np.mean(cos_vals))

    print("=" * 68)
    print(f"  {'Stride':>6}  {'ViT MSE':>10}  {'ViT Cos':>9}  "
          f"{'Compact MSE':>12}  {'Compact Cos':>11}")
    print("-" * 68)

    base_vit_mse = base_comp_mse = None

    for stride in range(1, max_stride + 1):
        vit_mse,  vit_cos  = compute_stats(vit_pairs[stride])
        comp_mse, comp_cos = compute_stats(comp_pairs[stride])

        if stride == 1:
            base_vit_mse  = vit_mse
            base_comp_mse = comp_mse
            suffix = "  ← baseline"
        else:
            vit_mult  = vit_mse  / base_vit_mse  if base_vit_mse  > 0 else 0
            comp_mult = comp_mse / base_comp_mse if base_comp_mse > 0 else 0
            suffix = f"  ViT ×{vit_mult:.2f}  Compact ×{comp_mult:.2f}"

        print(f"  {stride:>6}  {vit_mse:>10.6f}  {vit_cos:>9.4f}  "
              f"{comp_mse:>12.6f}  {comp_cos:>11.4f}{suffix}")

    print("=" * 68)
    print()
    print("Higher MSE / lower cos at larger stride → more signal for predictor.")


if __name__ == "__main__":
    main()
