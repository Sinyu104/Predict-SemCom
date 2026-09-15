"""
dataset.py  —  HDF5 trajectory dataset for the Predictive SemCom system.

HDF5 layouts supported (auto-detected)
---------------------------------------
Layout A — flat arrays (all episodes concatenated):
    /observations   (N, H, W, C)          uint8 or float32 RGB images
    /actions        (N, action_dim)        float32
    /poses          (N, pose_dim)          float32  [optional]
    /latents        (N, 4, H_lat, W_lat)   float32  [optional, pre-computed VAE latents]

Layout B — episodic groups (produced by isaac_collector.py):
    /episode_0/observations  (T, H, W, C)
    /episode_0/actions       (T, action_dim)
    /episode_0/poses         (T, pose_dim)          [optional]
    /episode_0/latents       (T, 4, H_lat, W_lat)   [optional, pre-computed VAE latents]
    /episode_1/...

Two dataset classes
--------------------
GNMTrajectoryDataset  — single-step pairs (obs_t, action_t, obs_tp1).
                        Also returns pre-computed latents if available.
                        Used by Stage 2 (JSCC) trainer.

ClipDataset           — T-frame clips for multi-frame predictor training.
                        Returns (frames, actions, poses, latents) where:
                          frames  : (T, C, H, W)        float32 [0,1]
                          actions : (T-1, action_dim)    float32
                          poses   : (T-1, pose_dim)      float32, or zeros if absent
                          latents : (T, 4, H_lat, W_lat) float32
                                    — real VAE latents if /latents present in HDF5,
                                      zeros otherwise (trainer falls back to VAE encoding)

Pre-computed latents
---------------------
Run precompute_latents.py once to add /latents to the HDF5.  Both dataset classes
detect the key automatically and set has_latents=True.  Stage 1/2 trainers use
this flag to skip the (expensive) online VAE encode step.

Notes
-----
• Images are always returned as (C, H, W) float32 normalised to [0, 1].
• Images are auto-resized to (obs_height, obs_width) if needed.
• Alpha channels are silently dropped.
• num_workers defaults to 0 to avoid h5py multiprocessing issues.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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


def _resolve_obs_key(grp, preferred: str | None = None) -> str | None:
    """
    Pick the observations dataset key inside an HDF5 group / file.

    Order: explicit `preferred` → "observations" → "observations_cam1" →
    "observations_cam2".  Returns None if none are present.
    """
    candidates = ([preferred] if preferred else []) + [
        "observations", "observations_cam1", "observations_cam2",
    ]
    for k in candidates:
        if k and k in grp:
            return k
    return None


def load_task_config(task_dir: str) -> dict:
    """Read task.json from a task data directory."""
    path = os.path.join(task_dir, "task.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"task.json not found at '{path}'")
    with open(path) as f:
        return json.load(f)


class GNMTrajectoryDataset(Dataset):
    """
    Single-step pairs (obs_t, action_t, obs_tp1) for Stage 2 (JSCC) training.

    If the HDF5 file contains pre-computed /latents (added by precompute_latents.py),
    the dataset sets has_latents=True and returns latent tensors alongside images.

    Returns
    -------
    obs_t    : (C, H, W)             float32
    action_t : (action_dim,)         float32
    obs_tp1  : (C, H, W)             float32
    lat_t    : (4, H_lat, W_lat)     float32  — real latent or zeros
    lat_tp1  : (4, H_lat, W_lat)     float32  — real latent or zeros
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

        self._index:      list = []
        self._action_dim: int | None = None
        self.has_latents: bool = False
        self._latent_shape: tuple = (4, obs_height // 8, obs_width // 8)
        self._load_index()

    def _load_index(self):
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"HDF5 file not found: '{self.hdf5_path}'. "
                "Run isaac_collector.py first or pass --stored_data."
            )

        with h5py.File(self.hdf5_path, "r") as f:
            if "observations" in f:
                n = len(f["observations"])
                self._index = [("flat", i) for i in range(n - 1)]
                if "actions" in f and f["actions"].ndim > 1:
                    self._action_dim = f["actions"].shape[1]
                if "latents" in f:
                    self.has_latents = True
                    self._latent_shape = f["latents"].shape[1:]   # (C, H, W)
            else:
                for ep_key in sorted(f.keys()):
                    grp = f[ep_key]
                    if "observations" not in grp or "actions" not in grp:
                        continue
                    n = len(grp["observations"])
                    for i in range(n - 1):
                        self._index.append((ep_key, i))
                    if self._action_dim is None and grp["actions"].ndim > 1:
                        self._action_dim = grp["actions"].shape[1]
                    if not self.has_latents and "latents" in grp:
                        self.has_latents = True
                        self._latent_shape = grp["latents"].shape[1:]

        if not self._index:
            raise ValueError(
                f"No valid triplets found in '{self.hdf5_path}'. "
                "Check the HDF5 layout (see dataset.py docstring)."
            )

    def _preprocess_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        return _preprocess_obs(obs_np, self.obs_height, self.obs_width)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        layout_key, i = self._index[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            if layout_key == "flat":
                obs_t    = np.array(f["observations"][i])
                obs_tp1  = np.array(f["observations"][i + 1])
                action_t = np.array(f["actions"][i], dtype=np.float32)
                if self.has_latents:
                    lat_t   = np.array(f["latents"][i],     dtype=np.float32)
                    lat_tp1 = np.array(f["latents"][i + 1], dtype=np.float32)
            else:
                grp      = f[layout_key]
                obs_t    = np.array(grp["observations"][i])
                obs_tp1  = np.array(grp["observations"][i + 1])
                action_t = np.array(grp["actions"][i], dtype=np.float32)
                if self.has_latents:
                    lat_t   = np.array(grp["latents"][i],     dtype=np.float32)
                    lat_tp1 = np.array(grp["latents"][i + 1], dtype=np.float32)

        obs_t   = self._preprocess_obs(obs_t)
        obs_tp1 = self._preprocess_obs(obs_tp1)

        if self.has_latents:
            return (obs_t, torch.from_numpy(action_t), obs_tp1,
                    torch.from_numpy(lat_t), torch.from_numpy(lat_tp1))
        else:
            zeros = torch.zeros(self._latent_shape)
            return obs_t, torch.from_numpy(action_t), obs_tp1, zeros, zeros


# ========================================================================== #
#  ClipDataset — T-frame clips for V-JEPA 2-AC Stage 1 training             #
# ========================================================================== #

class ClipDataset(Dataset):
    """
    T-frame clips for Stage 1 (world model) training.

    If the HDF5 file contains pre-computed /latents (added by precompute_latents.py),
    the dataset sets has_latents=True and returns them directly — the trainer then
    skips the online VAE encode step entirely.

    Returns
    -------
    frames  : (T, C, H, W)            float32 [0,1]
    actions : (T-1, action_dim)        float32
    poses   : (T-1, pose_dim)          float32  — zeros if /poses absent
    latents : (T, 4, H_lat, W_lat)     float32  — real latents or zeros
    """

    def __init__(
        self,
        hdf5_path:    str,
        obs_height:   int = 224,
        obs_width:    int = 224,
        obs_channels: int = 3,
        clip_length:  int = 16,
        stride:       int = 1,
        obs_key:      str | None = None,
        max_episodes: int | None = None,
    ):
        super().__init__()
        self.hdf5_path    = hdf5_path
        self.obs_height   = obs_height
        self.obs_width    = obs_width
        self.clip_length  = clip_length
        self.stride       = stride
        self.obs_key      = obs_key          # preferred camera key (e.g. "observations_cam1")
        self.max_episodes = max_episodes     # use only the first N episodes per file (None = all)
        self._obs_key:    str | None = None  # resolved at index time

        self._index:       list        = []
        self._action_dim:  int | None  = None
        self._pose_dim:    int         = 0
        self.has_latents:  bool        = False
        self._latent_shape: tuple      = (4, obs_height // 8, obs_width // 8)
        self._load_index()

    def _load_index(self):
        T      = self.clip_length
        stride = self.stride
        window = stride * (T - 1) + 1
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: '{self.hdf5_path}'")

        with h5py.File(self.hdf5_path, "r") as f:
            flat_key = _resolve_obs_key(f, self.obs_key)
            if flat_key is not None:
                self._obs_key = flat_key
                n = len(f[flat_key])
                self._index = [("flat", i) for i in range(n - window + 1)]
                if "actions" in f and f["actions"].ndim > 1:
                    self._action_dim = f["actions"].shape[1]
                if "poses" in f and f["poses"].ndim > 1:
                    self._pose_dim = f["poses"].shape[1]
                if "latents" in f:
                    self.has_latents   = True
                    self._latent_shape = f["latents"].shape[1:]
            else:
                # Numeric sort ("episode_2" < "episode_10") so a "first N" subset
                # is a contiguous, deterministic slice of the recorded demos.
                def _ep_num(k):
                    digits = "".join(ch for ch in k if ch.isdigit())
                    return int(digits) if digits else 0
                ep_keys = sorted(f.keys(), key=_ep_num)
                if self.max_episodes is not None:
                    ep_keys = ep_keys[: self.max_episodes]
                for ep_key in ep_keys:
                    grp = f[ep_key]
                    obs_key = _resolve_obs_key(grp, self.obs_key)
                    if obs_key is None or "actions" not in grp:
                        continue
                    if self._obs_key is None:
                        self._obs_key = obs_key
                    n = len(grp[obs_key])
                    if n < window:
                        continue
                    for i in range(n - window + 1):
                        self._index.append((ep_key, i))
                    if self._action_dim is None and grp["actions"].ndim > 1:
                        self._action_dim = grp["actions"].shape[1]
                    if self._pose_dim == 0 and "poses" in grp and grp["poses"].ndim > 1:
                        self._pose_dim = grp["poses"].shape[1]
                    if not self.has_latents and "latents" in grp:
                        self.has_latents   = True
                        self._latent_shape = grp["latents"].shape[1:]

        if not self._index:
            raise ValueError(
                f"No clips of length {T} (stride={stride}) found in '{self.hdf5_path}'. "
                "Check clip_length, clip_stride, or episode lengths."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        layout_key, start = self._index[idx]
        T             = self.clip_length
        stride        = self.stride
        frame_indices = [start + k * stride for k in range(T)]
        raw_end       = start + stride * (T - 1)

        with h5py.File(self.hdf5_path, "r") as f:
            if layout_key == "flat":
                obs_arr    = np.array(f[self._obs_key][frame_indices])
                action_raw = np.array(f["actions"][start:raw_end], dtype=np.float32)
                pose_arr   = np.array(f["poses"][frame_indices], dtype=np.float32) \
                             if "poses" in f else None
                lat_arr    = np.array(f["latents"][frame_indices], dtype=np.float32) \
                             if self.has_latents else None
            else:
                grp        = f[layout_key]
                obs_arr    = np.array(grp[self._obs_key][frame_indices])
                action_raw = np.array(grp["actions"][start:raw_end], dtype=np.float32)
                pose_arr   = np.array(grp["poses"][frame_indices], dtype=np.float32) \
                             if "poses" in grp else None
                lat_arr    = np.array(grp["latents"][frame_indices], dtype=np.float32) \
                             if self.has_latents else None

        action_dim = action_raw.shape[-1]
        action_arr = action_raw.reshape(T - 1, stride, action_dim).sum(axis=1)

        frames  = torch.stack([
            _preprocess_obs(obs_arr[k], self.obs_height, self.obs_width)
            for k in range(T)
        ])                                                              # (T, C, H, W)
        actions = torch.from_numpy(action_arr)                          # (T-1, D)
        poses   = torch.from_numpy(pose_arr[:-1]) if pose_arr is not None \
                  else torch.zeros(T - 1, max(self._pose_dim, 1))      # (T-1, pose_dim)
        latents = torch.from_numpy(lat_arr) if lat_arr is not None \
                  else torch.zeros(T, *self._latent_shape)              # (T, 4, H, W)

        return frames, actions, poses, latents


# ========================================================================== #
#  Multi-task dataset wrapper                                                #
# ========================================================================== #

class MultiTaskDataset(Dataset):
    """
    Wraps multiple GNMTrajectoryDataset instances (one per task) and tags
    each sample with the task's language instruction.

    __getitem__ returns (obs_t, action_t, obs_tp1, instruction).
    """

    def __init__(
        self,
        task_dirs:   list[str],
        obs_height:  int = 224,
        obs_width:   int = 224,
        obs_channels:int = 3,
    ):
        self._datasets     = []
        self._instructions = []
        self._offsets      = [0]

        for task_dir in task_dirs:
            cfg      = load_task_config(task_dir)
            hdf5_path= os.path.join(task_dir, "demos.hdf5")
            ds       = GNMTrajectoryDataset(hdf5_path, obs_height, obs_width, obs_channels)
            self._datasets.append(ds)
            self._instructions.append(cfg.get("instruction", ""))
            self._offsets.append(self._offsets[-1] + len(ds))

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int):
        # Find which sub-dataset this index falls into
        for i in range(len(self._datasets)):
            if idx < self._offsets[i + 1]:
                local_idx   = idx - self._offsets[i]
                obs_t, act, obs_tp1 = self._datasets[i][local_idx]
                return obs_t, act, obs_tp1, self._instructions[i]
        raise IndexError(f"Index {idx} out of range for MultiTaskDataset (len={len(self)})")


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


def build_multi_task_dataloaders(
    config:    dict,
    task_dirs: list[str],
) -> tuple:
    """
    Build train/val DataLoaders from multiple task directories.
    Each sample includes a language instruction string.

    Returns
    -------
    train_loader, val_loader  (batches are (obs_t, action_t, obs_tp1, instruction))
    """
    from torch.utils.data import random_split

    dataset = MultiTaskDataset(
        task_dirs    = task_dirs,
        obs_height   = config["obs_height"],
        obs_width    = config["obs_width"],
        obs_channels = config["obs_channels"],
    )

    n_total = len(dataset)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

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
        f"[dataset] multi-task  tasks={len(task_dirs)}  "
        f"total={n_total}  train={n_train}  val={n_val}"
    )
    return train_loader, val_loader
