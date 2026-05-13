"""
precompute_latents.py  —  Offline VAE encoding for Stage 1 training speedup.

Encodes all observation images in an HDF5 file using the frozen SD-VAE and
stores the resulting latents in a new HDF5 file (or in-place under a new key).

This removes the VAE encode() call from the training loop (~40% speedup because
VAE encoding is the bottleneck on CPU data-loading threads).

Usage
-----
  # Encode to a separate file
  python precompute_latents.py \\
      --input  data/trajectories.hdf5 \\
      --output data/trajectories_latents.hdf5 \\
      --vae    stabilityai/sd-vae-ft-mse \\
      --batch  64

  # In-place (adds /latents key alongside /observations)
  python precompute_latents.py \\
      --input  data/trajectories.hdf5 \\
      --inplace \\
      --vae    stabilityai/sd-vae-ft-mse

Output HDF5 layout
------------------
  Layout A (flat):
    /observations  (N, H, W, C)  uint8  [copied]
    /actions       (N, action_dim) float32  [copied]
    /latents       (N, 4, H_lat, W_lat) float32  [NEW: VAE latents]

  Layout B (episodic):
    /episode_k/observations  [copied]
    /episode_k/actions       [copied]
    /episode_k/latents       [NEW: VAE latents for each frame]
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm


# ========================================================================== #
#  VAE loading                                                                 #
# ========================================================================== #

def load_vae(model_name: str, device: torch.device) -> torch.nn.Module:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name).to(device)
    vae.eval()
    vae.requires_grad_(False)
    print(f"[precompute] Loaded VAE: {model_name}")
    return vae


@torch.no_grad()
def encode_batch(
    vae:    torch.nn.Module,
    imgs:   np.ndarray,     # (N, H, W, C) uint8
    height: int,
    width:  int,
    device: torch.device,
) -> np.ndarray:
    """
    Encode a batch of uint8 images → float32 VAE latents.
    Returns (N, 4, H_lat, W_lat) float32 numpy array.
    """
    N = imgs.shape[0]
    tensors = []
    for i in range(N):
        img = imgs[i].astype(np.float32) / 255.0          # (H, W, C) [0,1]
        if img.shape[2] == 4:
            img = img[..., :3]
        t = torch.from_numpy(img).permute(2, 0, 1)        # (C, H, W)
        if t.shape[1] != height or t.shape[2] != width:
            t = TF.resize(t, [height, width],
                          interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        # Normalize to [-1, 1] for VAE
        t = t * 2.0 - 1.0
        tensors.append(t)

    batch = torch.stack(tensors, dim=0).to(device)         # (N, C, H, W)
    latents = vae.encode(batch).latent_dist.mean            # (N, 4, H_lat, W_lat)
    latents = latents * vae.config.scaling_factor
    return latents.cpu().numpy().astype(np.float32)


# ========================================================================== #
#  Main                                                                        #
# ========================================================================== #

def process_flat(
    src:    h5py.File,
    dst:    h5py.File,
    vae:    torch.nn.Module,
    height: int,
    width:  int,
    device: torch.device,
    batch:  int,
):
    """Process Layout A (flat /observations array)."""
    obs  = src["observations"]
    N    = len(obs)
    H_lat = height // 8
    W_lat = width  // 8

    # Create latents dataset
    lat_ds = dst.require_dataset(
        "latents",
        shape=(N, 4, H_lat, W_lat),
        dtype=np.float32,
        chunks=(min(batch, N), 4, H_lat, W_lat),
    )

    for start in tqdm(range(0, N, batch), desc="  flat", unit="batch"):
        end  = min(start + batch, N)
        imgs = obs[start:end]
        lats = encode_batch(vae, imgs, height, width, device)
        lat_ds[start:end] = lats

    print(f"  [precompute] flat: {N} frames → latents {lat_ds.shape}")


def process_episodic(
    src:    h5py.File,
    dst:    h5py.File,
    vae:    torch.nn.Module,
    height: int,
    width:  int,
    device: torch.device,
    batch:  int,
):
    """Process Layout B (episodic /episode_k groups)."""
    H_lat = height // 8
    W_lat = width  // 8

    ep_keys = sorted(k for k in src.keys() if "observations" in src[k])
    for ep_key in tqdm(ep_keys, desc="  episodes"):
        grp_src = src[ep_key]
        grp_dst = dst.require_group(ep_key)

        obs = grp_src["observations"]
        N   = len(obs)

        lat_ds = grp_dst.require_dataset(
            "latents",
            shape=(N, 4, H_lat, W_lat),
            dtype=np.float32,
            chunks=(min(batch, N), 4, H_lat, W_lat),
        )

        for start in range(0, N, batch):
            end  = min(start + batch, N)
            imgs = obs[start:end]
            lats = encode_batch(vae, imgs, height, width, device)
            lat_ds[start:end] = lats


def copy_metadata(src: h5py.File, dst: h5py.File):
    """Copy all datasets except latents (already handled)."""
    def _copy(name, obj):
        if isinstance(obj, h5py.Dataset) and name != "latents" and not name.endswith("/latents"):
            src.copy(name, dst, name)

    src.visititems(_copy)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute SD-VAE latents for SemCom training.")
    parser.add_argument("--input",   required=True,  help="Source HDF5 file.")
    parser.add_argument("--output",  default=None,   help="Output HDF5 file. Defaults to input + '_latents'.")
    parser.add_argument("--inplace", action="store_true", help="Add latents key in-place to input file.")
    parser.add_argument("--vae",     default="stabilityai/sd-vae-ft-mse", help="VAE model name or path.")
    parser.add_argument("--height",  type=int, default=224)
    parser.add_argument("--width",   type=int, default=224)
    parser.add_argument("--batch",   type=int, default=64,  help="Encode batch size.")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[precompute] ERROR: input file not found: {args.input}")
        sys.exit(1)

    device = torch.device(args.device)
    vae    = load_vae(args.vae, device)

    if args.inplace:
        out_path = args.input
        mode     = "r+"
    else:
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = base + "_latents" + ext
        out_path = args.output
        mode     = "w"

    print(f"[precompute] {args.input}  →  {out_path}")
    print(f"[precompute] VAE: {args.vae}  |  img size: {args.height}×{args.width}  |  batch: {args.batch}")

    with h5py.File(args.input, "r") as src, h5py.File(out_path, mode) as dst:
        # Detect layout
        if "observations" in src:
            layout = "flat"
        else:
            layout = "episodic"
        print(f"[precompute] Detected layout: {layout}")

        if not args.inplace:
            print("[precompute] Copying source datasets …")
            copy_metadata(src, dst)

        if layout == "flat":
            process_flat(src, dst, vae, args.height, args.width, device, args.batch)
        else:
            process_episodic(src, dst, vae, args.height, args.width, device, args.batch)

    print(f"[precompute] Done → {out_path}")
    print(f"[precompute] Latent size per frame: 4 × {args.height//8} × {args.width//8}")


if __name__ == "__main__":
    main()
