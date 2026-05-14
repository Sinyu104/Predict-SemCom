"""
quick_vis.py  —  Inspect an HDF5 demo file without launching Isaac Sim.
Run with any Python that has h5py + matplotlib installed.

Usage:
  python quick_vis.py                              # default file, first 4 episodes, all cameras
  python quick_vis.py --hdf5 path/to/file.hdf5 --episodes 0 2 5 --frames 8
  python quick_vis.py --episodes all --camera 2    # only show camera 2
  python quick_vis.py --episodes all               # every episode in the file
"""
import argparse
import os
import sys

import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",     type=str, default=r"data\pick_yellow_cube_to_tray\demos.hdf5")
    p.add_argument("--episodes", nargs="+", default=["0", "1", "2", "3"],
                   help="Episode indices to visualise, or 'all'")
    p.add_argument("--frames",   type=int, default=6,
                   help="Number of evenly-spaced frames to show per episode")
    p.add_argument("--camera",   nargs="+", type=int, default=None,
                   help="Camera IDs to show (default: all found in file). E.g. --camera 2 3")
    p.add_argument("--out",      type=str, default=None,
                   help="Output PNG path (default: next to the HDF5 file)")
    return p.parse_args()

args = parse_args()

try:
    import h5py
except ImportError:
    sys.exit("Install h5py:  pip install h5py")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Install matplotlib:  pip install matplotlib")


def _obs_keys(ep_grp) -> list[str]:
    """Return sorted observation dataset keys found in an episode group."""
    # New format: observations_cam1, observations_cam2, observations_cam3
    cam_keys = sorted([k for k in ep_grp.keys() if k.startswith("observations_cam")])
    if cam_keys:
        return cam_keys
    # Old format: single 'observations' dataset
    if "observations" in ep_grp:
        return ["observations"]
    return []


with h5py.File(args.hdf5, "r") as f:
    all_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
    print(f"Total episodes in file: {len(all_keys)}")

    # Resolve which episodes to show
    if args.episodes == ["all"]:
        ep_keys = all_keys
    else:
        ep_keys = []
        for e in args.episodes:
            idx = int(e)
            if idx < len(all_keys):
                ep_keys.append(all_keys[idx])
            else:
                print(f"  [warn] episode {idx} not found (file has {len(all_keys)} episodes)")

    if not ep_keys:
        sys.exit("No valid episodes selected.")

    # Discover all observation keys from the first episode
    all_obs_keys = _obs_keys(f[ep_keys[0]])
    if not all_obs_keys:
        sys.exit("No observation datasets found in this file.")

    # Filter by --camera if specified
    if args.camera is not None:
        wanted = {f"observations_cam{c}" for c in args.camera} | ({"observations"} if 1 in args.camera else set())
        all_obs_keys = [k for k in all_obs_keys if k in wanted]
        if not all_obs_keys:
            sys.exit(f"None of the requested cameras found. Available: {_obs_keys(f[ep_keys[0]])}")

    n_eps     = len(ep_keys)
    n_cams    = len(all_obs_keys)
    n_frames  = args.frames
    n_rows    = n_eps * n_cams   # one row per (episode, camera) pair

    fig, axes = plt.subplots(n_rows, n_frames,
                             figsize=(n_frames * 3, n_rows * 2.5),
                             squeeze=False)

    row = 0
    for ep_key in ep_keys:
        ep_grp  = f[ep_key]
        act     = ep_grp["actions"][:]
        success = ep_grp.attrs.get("success", "?")
        steps   = ep_grp.attrs.get("steps", "?")

        for obs_key in all_obs_keys:
            if obs_key not in ep_grp:
                print(f"  [warn] {ep_key}/{obs_key} missing — skipping")
                row += 1
                continue

            obs = ep_grp[obs_key][:]   # (T, H, W, 3)
            T   = obs.shape[0]
            cam_label = obs_key.replace("observations_", "").replace("observations", "cam1")

            print(f"  {ep_key}/{obs_key}:  shape={obs.shape} dtype={obs.dtype} "
                  f"min={obs.min()} max={obs.max()}  success={success}")

            frame_indices = np.linspace(0, T - 1, min(n_frames, T), dtype=int)
            for col, idx in enumerate(frame_indices):
                ax = axes[row][col]
                ax.imshow(obs[idx])
                ax.set_title(f"step {idx}", fontsize=7)
                ax.axis("off")

            axes[row][0].set_ylabel(
                f"{ep_key}\n{cam_label}\nsuccess={success} steps={steps}",
                fontsize=7, rotation=0, labelpad=70, va="center"
            )
            row += 1

    plt.suptitle(
        f"{os.path.basename(args.hdf5)}  —  {n_eps} ep(s)  ×  {n_cams} cam(s)  ×  {n_frames} frames",
        fontsize=11
    )
    plt.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.hdf5)), "quick_vis_frames.png"
    )
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {out}")
