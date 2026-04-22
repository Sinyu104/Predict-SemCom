"""
dataset.py  —  HDF5 trajectory dataset for the Predictive SemCom system.

HDF5 layouts supported (auto-detected)
---------------------------------------
Layout A — flat arrays (all episodes concatenated):
    /observations   (N, H, W, C)    uint8 or float32 RGB images
    /actions        (N, action_dim)  float32
    /poses          (N, pose_dim)    float32  [optional]

Layout B — episodic groups (produced by isaac_collector.py):
    /episode_0/observations  (T, H, W, C)
    /episode_0/actions       (T, action_dim)
    /episode_0/poses         (T, pose_dim)   [optional]
    /episode_1/...

Two dataset classes
--------------------
GNMTrajectoryDataset  — single-step pairs (obs_t, action_t, obs_tp1).
                        Used by Stage 2 (JSCC) trainer.

ClipDataset           — T-frame clips for multi-frame predictor training
                        (V-JEPA 2-AC, paper §3.1).
                        Returns (frames, actions, poses) where:
                          frames  : (T, C, H, W)  float32 [0,1]
                          actions : (T, action_dim) float32 — a_k at step k;
                                    a_T = 0 (dummy, padded internally)
                          poses   : (T, pose_dim) float32, or zeros if absent

Notes
-----
• Images are always returned as (C, H, W) float32 normalised to [0, 1].
• Images are auto-resized to (obs_height, obs_width) if needed.
• Alpha channels are silently dropped.
• num_workers defaults to 0 to avoid h5py multiprocessing issues.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import h5py


def _preprocess_obs(
    obs_np:      np.ndarray,
    obs_height:  int,
    obs_width:   int,
) -> torch.Tensor:
    """(H,W,C) or (C,H,W) uint8/float32 → (C,H,W) float32 [0,1], resized."""
    if obs_np.ndim == 3 and obs_np.shape[0] in (1, 3, 4):
        obs_np = obs_np.transpose(1, 2, 0)
    obs_np = obs_np.astype(np.float32) / 255.0 if obs_np.dtype == np.uint8 \
             else np.clip(obs_np.astype(np.float32), 0.0, 1.0)
    if obs_np.ndim == 3 and obs_np.shape[2] == 4:
        obs_np = obs_np[..., :3]
    t = torch.from_numpy(obs_np).permute(2, 0, 1).contiguous()
    if t.shape[1] != obs_height or t.shape[2] != obs_width:
        t = TF.resize(t, [obs_height, obs_width],
                      interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return t


class GNMTrajectoryDataset(Dataset):
    """
    Parameters
    ----------
    hdf5_path    : str   Path to the .hdf5 file.
    obs_height   : int   Target image height after optional resize.
    obs_width    : int   Target image width after optional resize.
    obs_channels : int   Expected colour channels (default 3 = RGB).
    """

    def __init__(
        self,
        hdf5_path:    str,
        obs_height:   int = 224,
        obs_width:    int = 224,
        obs_channels: int = 3,
    ):
        super().__init__()
        self.hdf5_path    = hdf5_path
        self.obs_height   = obs_height
        self.obs_width    = obs_width
        self.obs_channels = obs_channels

        self._index: list = []          # list of (layout_key, step_index)
        self._action_dim  = None        # detected from file
        self._load_index()

    # ------------------------------------------------------------------ #
    #  Index construction                                                  #
    # ------------------------------------------------------------------ #

    def _load_index(self):
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"HDF5 file not found: '{self.hdf5_path}'. "
                "Run isaac_collector.py first or pass --stored_data."
            )

        with h5py.File(self.hdf5_path, "r") as f:
            if "observations" in f:
                # Layout A: flat arrays
                n = len(f["observations"])
                self._index = [("flat", i) for i in range(n - 1)]
                if "actions" in f:
                    shape = f["actions"].shape
                    if len(shape) > 1:
                        self._action_dim = shape[1]
            else:
                # Layout B: episodic groups
                for ep_key in sorted(f.keys()):
                    grp = f[ep_key]
                    if "observations" not in grp or "actions" not in grp:
                        continue
                    n = len(grp["observations"])
                    for i in range(n - 1):
                        self._index.append((ep_key, i))
                    if self._action_dim is None:
                        shape = grp["actions"].shape
                        if len(shape) > 1:
                            self._action_dim = shape[1]

        if len(self._index) == 0:
            raise ValueError(
                f"No valid triplets found in '{self.hdf5_path}'. "
                "Check the HDF5 layout (see dataset.py docstring)."
            )

    # ------------------------------------------------------------------ #
    #  Image preprocessing                                                 #
    # ------------------------------------------------------------------ #

    def _preprocess_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        return _preprocess_obs(obs_np, self.obs_height, self.obs_width)

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        obs_t    : (C, H, W)     float32
        action_t : (action_dim,) float32
        obs_tp1  : (C, H, W)     float32
        """
        layout_key, i = self._index[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            if layout_key == "flat":
                obs_t    = np.array(f["observations"][i])
                obs_tp1  = np.array(f["observations"][i + 1])
                action_t = np.array(f["actions"][i], dtype=np.float32)
            else:
                grp      = f[layout_key]
                obs_t    = np.array(grp["observations"][i])
                obs_tp1  = np.array(grp["observations"][i + 1])
                action_t = np.array(grp["actions"][i], dtype=np.float32)

        obs_t   = self._preprocess_obs(obs_t)
        obs_tp1 = self._preprocess_obs(obs_tp1)

        return obs_t, torch.from_numpy(action_t), obs_tp1


# ========================================================================== #
#  ClipDataset — T-frame clips for V-JEPA 2-AC Stage 1 training             #
# ========================================================================== #

class ClipDataset(Dataset):
    """
    Returns T-frame clips sampled from the same episode.

    Returns
    -------
    frames  : (T, C, H, W)      float32 [0,1]
    actions : (T, action_dim)   float32  — a_k at step k (clip has T steps)
    poses   : (T, pose_dim)     float32  — zeros if /poses absent in HDF5
    """

    def __init__(
        self,
        hdf5_path:    str,
        obs_height:   int = 224,
        obs_width:    int = 224,
        obs_channels: int = 3,
        clip_length:  int = 16,
    ):
        super().__init__()
        self.hdf5_path   = hdf5_path
        self.obs_height  = obs_height
        self.obs_width   = obs_width
        self.clip_length = clip_length

        self._index      = []   # (layout_key, start_idx)
        self._action_dim = None
        self._pose_dim   = 0
        self._load_index()

    def _load_index(self):
        T = self.clip_length
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: '{self.hdf5_path}'")

        with h5py.File(self.hdf5_path, "r") as f:
            if "observations" in f:
                n = len(f["observations"])
                self._index = [("flat", i) for i in range(n - T + 1)]
                if "actions" in f and f["actions"].ndim > 1:
                    self._action_dim = f["actions"].shape[1]
                if "poses" in f and f["poses"].ndim > 1:
                    self._pose_dim = f["poses"].shape[1]
            else:
                for ep_key in sorted(f.keys()):
                    grp = f[ep_key]
                    if "observations" not in grp or "actions" not in grp:
                        continue
                    n = len(grp["observations"])
                    if n < T:
                        continue
                    for i in range(n - T + 1):
                        self._index.append((ep_key, i))
                    if self._action_dim is None and grp["actions"].ndim > 1:
                        self._action_dim = grp["actions"].shape[1]
                    if self._pose_dim == 0 and "poses" in grp and grp["poses"].ndim > 1:
                        self._pose_dim = grp["poses"].shape[1]

        if not self._index:
            raise ValueError(
                f"No clips of length {T} found in '{self.hdf5_path}'. "
                "Check clip_length or episode lengths."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        layout_key, start = self._index[idx]
        T = self.clip_length
        end = start + T

        with h5py.File(self.hdf5_path, "r") as f:
            if layout_key == "flat":
                obs_arr    = np.array(f["observations"][start:end])
                action_arr = np.array(f["actions"][start:end], dtype=np.float32)
                pose_arr   = np.array(f["poses"][start:end], dtype=np.float32) \
                             if "poses" in f else None
            else:
                grp        = f[layout_key]
                obs_arr    = np.array(grp["observations"][start:end])
                action_arr = np.array(grp["actions"][start:end], dtype=np.float32)
                pose_arr   = np.array(grp["poses"][start:end], dtype=np.float32) \
                             if "poses" in grp else None

        frames  = torch.stack([
            _preprocess_obs(obs_arr[k], self.obs_height, self.obs_width)
            for k in range(T)
        ])                                                    # (T, C, H, W)
        actions = torch.from_numpy(action_arr)                # (T, action_dim)
        poses   = torch.from_numpy(pose_arr) if pose_arr is not None \
                  else torch.zeros(T, max(self._pose_dim, 1)) # (T, pose_dim) or zeros

        return frames, actions, poses


# ========================================================================== #
#  DataLoader builder                                                        #
# ========================================================================== #

def build_dataloaders(
    config:    dict,
    hdf5_path: str,
) -> tuple:
    """
    80/20 train/val split with deterministic seed.
    Validates action_dim against config before building loaders.

    Returns
    -------
    train_loader, val_loader
    """
    from torch.utils.data import random_split

    dataset = GNMTrajectoryDataset(
        hdf5_path    = hdf5_path,
        obs_height   = config["obs_height"],
        obs_width    = config["obs_width"],
        obs_channels = config["obs_channels"],
    )

    # Validate action dimension
    if dataset._action_dim is not None:
        expected = config.get("action_dim", 7)
        if dataset._action_dim != expected:
            raise ValueError(
                f"Action dimension mismatch: "
                f"HDF5 has action_dim={dataset._action_dim} but "
                f"config['action_dim']={expected}. "
                "Update config.py or re-collect the dataset."
            )

    n_total = len(dataset)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=generator
    )

    # Use num_workers=0 by default to avoid h5py + fork issues.
    # Set to 4+ only when h5py is built with SWMR / thread-safe HDF5.
    num_workers = config.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size         = config["batch_size"],
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = config["batch_size"],
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = True,
        drop_last          = False,
        persistent_workers = (num_workers > 0),
    )

    print(
        f"[dataset] '{os.path.basename(hdf5_path)}'  "
        f"total={n_total}  train={n_train}  val={n_val}  "
        f"action_dim={dataset._action_dim}"
    )
    return train_loader, val_loader
