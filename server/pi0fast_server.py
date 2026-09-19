"""
pi0fast_server.py  —  PI0Fast fine-tuning on Franka demonstrations.

Mirrors the structure of vla_server.py but targets lerobot's PI0FastPolicy
instead of OpenVLA.

HDF5 layout (data/<task>/demos.hdf5):
    /episode_N/observations_cam1  (T, 224, 224, 3) uint8
    /episode_N/observations_cam2  (T, 224, 224, 3) uint8
    /episode_N/actions            (T, 7)           float32

Usage
-----
Multi-task fine-tune (all tasks):
    conda activate worldmodel
    python server/pi0fast_server.py \\
        --model_dir /path/to/pi0-fast \\
        --tasks all \\
        --finetune_output outputs/pi0fast_finetuned \\
        --epochs 20 --batch_size 4

Multi-task fine-tune (specific tasks):
    python server/pi0fast_server.py \\
        --model_dir /path/to/pi0-fast \\
        --tasks pick_red_cube_to_tray pick_blue_cube_to_tray \\
        --finetune_output outputs/pi0fast_finetuned

Single-task fine-tune:
    python server/pi0fast_server.py \\
        --model_dir /path/to/pi0-fast \\
        --demo_data data/pick_blue_cube_to_tray/demos.hdf5 \\
        --finetune_output outputs/pi0fast_finetuned \\
        --instruction "pick up the blue cube and place it on the tray"

Fine-tune (action-expert only — less VRAM):
    python server/pi0fast_server.py \\
        --model_dir /path/to/pi0-fast \\
        --tasks all \\
        --finetune_output outputs/pi0fast_finetuned \\
        --freeze_vision --freeze_language

Full fine-tune (all params — like `lerobot-train --policy.type=pi0_fast`):
    python server/pi0fast_server.py \\
        --model_dir /path/to/pi0-fast \\
        --tasks all \\
        --finetune_output outputs/pi0fast_full_finetuned \\
        --full_finetune
    # Needs much more VRAM/host RAM than LoRA (fp32 AdamW: params + 2 moments,
    # ~3x model size, sharded via FSDP). Checkpoints write full_weights.pt.

Resume from checkpoint:
    python server/pi0fast_server.py \\
        --resume_from outputs/pi0fast_finetuned/checkpoint_epoch5 \\
        --tasks all \\
        --finetune_output outputs/pi0fast_finetuned

Camera keys in the batch default to:
    observation.images.cam1   ←  observations_cam1
    observation.images.cam2   ←  observations_cam2
Override with --cam1_key / --cam2_key to match the pretrained model's config.
"""

import argparse
import json
import os
import re
import shutil
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm


# ── Silence lerobot's broken groot module before any lerobot.policies import ─ #

def _mock_groot():
    from unittest.mock import MagicMock
    for mod in [
        "lerobot.policies.groot",
        "lerobot.policies.groot.modeling_groot",
        "lerobot.policies.groot.configuration_groot",
        "lerobot.policies.groot.groot_n1",
    ]:
        sys.modules.setdefault(mod, MagicMock())


# ========================================================================== #
#  Action-chunk tokeniser helper                                              #
# ========================================================================== #

def tokenize_action_chunks(
    action_chunks: torch.Tensor,   # (B, K, action_dim) float32
    policy,
    action_low=None,               # (action_dim,) per-dim min, or None
    action_high=None,              # (action_dim,) per-dim max, or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert continuous action chunks to FAST token IDs using the tokenisers
    already loaded inside the policy.

    Replicates ActionTokenizerProcessorStep._tokenize_action without the
    full lerobot processor pipeline.

    If action_low/action_high are given, the real action dims are first
    normalised per-dim to [-1, 1] (affine min->-1, max->+1).  This is REQUIRED:
    the FAST tokenizer expects normalised actions; feeding raw values (e.g.
    SO-101 degrees, range ~-80..102) overflows its representable DC range and
    floors anything below ~-6.46, silently destroying strongly-negative targets
    (measured: shoulder_lift/wrist_roll saturated).  Inference must denormalise
    with the SAME stats, which are persisted alongside the checkpoint.

    Returns
    -------
    tokens : (B, max_action_tokens) long
    mask   : (B, max_action_tokens) bool  — True = real token
    """
    max_tokens = policy.config.max_action_tokens
    fast_skip  = policy.config.fast_skip_tokens
    pali_tok   = policy._paligemma_tokenizer
    fast_tok   = policy.action_tokenizer   # AutoProcessor from lerobot/fast-action-tokenizer

    bos_id        = pali_tok.bos_token_id
    action_prefix = torch.tensor(
        pali_tok.encode("Action: ", add_special_tokens=False), dtype=torch.long)
    end_token     = torch.tensor(
        pali_tok.encode("|"), dtype=torch.long)

    # Per-dim affine normalise to [-1, 1] on the REAL dims, BEFORE zero-padding
    # to max_action_dim (the padding dims stay 0 = the [-1,1] midpoint).
    if action_low is not None and action_high is not None:
        lo = torch.as_tensor(action_low, dtype=action_chunks.dtype, device=action_chunks.device)
        hi = torch.as_tensor(action_high, dtype=action_chunks.dtype, device=action_chunks.device)
        action_chunks = 2.0 * (action_chunks - lo) / (hi - lo) - 1.0

    # pi0-fast was trained with actions padded to max_action_dim=32.
    # Tokenizing raw 7-DoF actions produces wrong token IDs vs training distribution.
    max_action_dim = getattr(policy.config, "max_action_dim", 32)
    if action_chunks.shape[-1] < max_action_dim:
        action_chunks = F.pad(
            action_chunks, (0, max_action_dim - action_chunks.shape[-1])
        )  # (B, K, max_action_dim)

    tokens_list, masks_list = [], []
    for i in range(action_chunks.size(0)):
        act_cpu = action_chunks[i : i + 1].cpu()   # (1, K, max_action_dim)
        raw     = fast_tok(act_cpu)                 # list or Tensor

        if not isinstance(raw, torch.Tensor):
            raw = torch.tensor(raw, dtype=torch.long)
        if raw.dim() > 1:
            raw = raw.flatten()

        # Map FAST token IDs → PaliGemma vocabulary space (clamp to prevent neg IDs)
        pali_ids = (pali_tok.vocab_size - 1 - fast_skip - raw.long()).clamp(min=0)

        tok = torch.cat([
            torch.tensor([bos_id], dtype=torch.long),
            action_prefix,
            pali_ids,
            end_token,
        ])

        if len(tok) >= max_tokens:
            tok  = tok[:max_tokens]
            mask = torch.ones(max_tokens, dtype=torch.bool)
        else:
            pad  = max_tokens - len(tok)
            mask = torch.cat([torch.ones(len(tok), dtype=torch.bool),
                               torch.zeros(pad,    dtype=torch.bool)])
            tok  = F.pad(tok, (0, pad), value=0)

        tokens_list.append(tok)
        masks_list.append(mask)

    return torch.stack(tokens_list), torch.stack(masks_list)


# ========================================================================== #
#  Action normalisation stats                                                 #
# ========================================================================== #

def compute_action_stats(dataset):
    """Per-dim [min, max] over ALL actions in the dataset — the FULL range, not a
    per-epoch subset. Works for DemoDataset (actions held in RAM) and
    MultiTaskDemoDataset (scans every episode's actions from disk). Constant dims
    (max==min) get their range widened to 1 to avoid a divide-by-zero at
    normalise time. Returns (low, high) float32 arrays of length action_dim.
    """
    lo = hi = None

    def _upd(a):
        nonlocal lo, hi
        a = np.asarray(a, dtype=np.float64)
        amin, amax = a.min(axis=0), a.max(axis=0)
        lo = amin if lo is None else np.minimum(lo, amin)
        hi = amax if hi is None else np.maximum(hi, amax)

    if getattr(dataset, "_acts", None):                       # DemoDataset (in RAM)
        for a in dataset._acts:
            _upd(a)
    elif hasattr(dataset, "_task_meta"):                      # MultiTaskDemoDataset
        for hdf5_path, _instr, ep_keys, _lens in dataset._task_meta:
            with h5py.File(hdf5_path, "r") as f:
                for k in ep_keys:
                    _upd(f[k]["actions"])
    else:
        raise ValueError("compute_action_stats: unsupported dataset type "
                         f"{type(dataset).__name__}")

    rng = hi - lo
    hi = np.where(rng < 1e-6, lo + 1.0, hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def save_norm_stats(out_dir, action_low, action_high, action_order=None):
    """Persist normalisation stats next to the checkpoint as norm_stats.json, so
    inference (vla_server.py) denormalises with EXACTLY these numbers rather than
    recomputing or hardcoding — a train/infer mismatch reintroduces a silent
    per-dim offset."""
    stats = {
        "action_low":  [float(x) for x in action_low],
        "action_high": [float(x) for x in action_high],
        "normalization": "affine_per_dim_[-1,1]",
    }
    if action_order is not None:
        stats["action_order"] = list(action_order)
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


# ========================================================================== #
#  Language token helper                                                      #
# ========================================================================== #

def build_language_tokens(
    instruction: str,
    state_dim:   int,
    policy,
    batch_size:  int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenise "Task: <instruction>, State: <zeros>;\n" once and expand to batch.

    Returns (ids, mask) each (B, tokenizer_max_length) long.
    """
    tok_max  = getattr(policy.config, "tokenizer_max_length", 200)
    pali_tok = policy._paligemma_tokenizer

    # Zero state → bin 128 (midpoint of 256 bins over [-1, 1])
    disc      = np.full(state_dim, 128, dtype=int)
    state_str = " ".join(map(str, disc))
    cleaned   = instruction.strip().replace("_", " ").replace("\n", " ")
    prompt    = f"Task: {cleaned}, State: {state_str};\n"

    enc  = pali_tok(prompt, return_tensors="pt", padding="max_length",
                    max_length=tok_max, truncation=True)
    ids  = enc["input_ids"].expand(batch_size, -1).to(device)
    mask = enc["attention_mask"].expand(batch_size, -1).to(device)
    return ids, mask


def build_language_tokens_batch(
    instructions: list[str],
    state_dim:    int,
    policy,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenise a list of (possibly different) instructions and stack into a batch.

    Used in multi-task training where each sample has its own instruction.
    Returns (ids, mask) each (B, tokenizer_max_length) long.
    """
    tok_max  = getattr(policy.config, "tokenizer_max_length", 200)
    pali_tok = policy._paligemma_tokenizer
    disc     = np.full(state_dim, 128, dtype=int)
    state_str = " ".join(map(str, disc))

    ids_list, mask_list = [], []
    for instr in instructions:
        cleaned = instr.strip().replace("_", " ").replace("\n", " ")
        prompt  = f"Task: {cleaned}, State: {state_str};\n"
        enc = pali_tok(prompt, return_tensors="pt", padding="max_length",
                       max_length=tok_max, truncation=True)
        ids_list.append(enc["input_ids"])
        mask_list.append(enc["attention_mask"])

    ids  = torch.cat(ids_list,  dim=0).to(device)   # (B, tok_max)
    mask = torch.cat(mask_list, dim=0).to(device)
    return ids, mask


# ========================================================================== #
#  HDF5 demonstration dataset (dual-camera)                                   #
# ========================================================================== #

class DemoDataset(TorchDataset):
    """
    Loads Franka dual-camera demos from demos.hdf5 and yields
    (cam1_img, cam2_img, action_chunk) tuples.

    HDF5 layout:
        /episode_N/observations_cam1  (T, 224, 224, 3) uint8
        /episode_N/observations_cam2  (T, 224, 224, 3) uint8
        /episode_N/actions            (T, 7)           float32

    Each sample is a window starting at frame t:
        cam1_img   : (3, 224, 224) float [0, 1]
        cam2_img   : (3, 224, 224) float [0, 1]
        act_chunk  : (chunk_size, 7) float32
    """

    def __init__(
        self,
        path:         str,
        chunk_size:   int = 50,
        frame_stride: int = 1,
    ):
        self.chunk_size = chunk_size
        self._cam1 = []   # list of (T, H, W, 3) uint8 per episode
        self._cam2 = []
        self._acts = []   # list of (T, 7) float32 per episode
        self.samples = [] # list of (ep_idx, t)

        print(f"[DemoDataset] Loading {path} …")
        with h5py.File(path, "r") as f:
            ep_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
            for ep_key in ep_keys:
                grp = f[ep_key]
                if "observations_cam1" not in grp or "actions" not in grp:
                    continue
                self._cam1.append(np.array(grp["observations_cam1"]))  # (T,224,224,3)
                self._cam2.append(np.array(grp["observations_cam2"]))
                self._acts.append(np.array(grp["actions"], dtype=np.float32))

        n_eps = len(self._acts)
        for ep_idx, acts in enumerate(self._acts):
            T = len(acts)
            for t in range(0, T - chunk_size, frame_stride):
                self.samples.append((ep_idx, t))

        print(f"[DemoDataset] {n_eps} episodes → {len(self.samples)} samples "
              f"(chunk={chunk_size}, stride={frame_stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]

        def to_tensor(arr):
            # arr: (H, W, 3) uint8 → (3, H, W) float [0, 1]
            return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        cam1 = to_tensor(self._cam1[ep_idx][t])   # (3, 224, 224)
        cam2 = to_tensor(self._cam2[ep_idx][t])

        chunk = self._acts[ep_idx][t : t + self.chunk_size]  # (K, 7)
        if len(chunk) < self.chunk_size:
            pad   = np.zeros((self.chunk_size - len(chunk), chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        act_chunk = torch.from_numpy(chunk)                   # (K, 7)

        return cam1, cam2, act_chunk


ALL_TASKS = [
    "pick_red_cube_to_tray",
    "pick_blue_cube_to_tray",
    "pick_yellow_cube_to_tray",
    "pick_orange_cube_to_tray",
    "pick_purple_cube_to_tray",
    "pick_red_cube_to_front_left_corner",
    "pick_red_cube_to_front_right_corner",
    "pick_red_cube_to_back_left_corner",
    "pick_red_cube_to_back_right_corner",
    "pick_leftmost_cube",
    "stack_red_on_blue",
    "stack_blue_on_red",
    "sort_two_cubes",
]


class MultiTaskDemoDataset(torch.utils.data.Dataset):
    """
    Lazy-loading multi-task dataset.

    At init: scans HDF5 metadata only (no pixel data read into RAM).
    Each epoch: call resample(epoch) to randomly pick n_episodes_per_task
                episodes per task — different episodes each epoch.
    __getitem__: reads only the requested frame directly from HDF5 on disk.

    RAM usage: ~zero for data (only frame index metadata).
    """

    def __init__(
        self,
        task_dirs:          list[str],
        chunk_size:         int = 30,
        n_episodes_per_task: int = 25,
        frame_stride:       int = 1,
        val_split:          float = 0.0,
        val_seed:           int = 0,
    ):
        self.chunk_size          = chunk_size
        self.n_episodes_per_task = n_episodes_per_task
        self.frame_stride        = frame_stride
        self.val_split           = val_split

        # Metadata only — no pixel arrays loaded
        self._task_meta  = []   # list of (hdf5_path, instruction, train_ep_keys, ep_lengths)
        self._handles    = {}   # hdf5_path → open h5py.File (opened lazily)
        self.samples     = []   # populated by resample() — training samples only
        self.val_samples = []   # fixed held-out samples, built once here

        for task_dir in task_dirs:
            hdf5_path = os.path.join(task_dir, "demos.hdf5")
            json_path = os.path.join(task_dir, "task.json")
            if not os.path.isfile(hdf5_path):
                print(f"[MultiTaskDemoDataset] Skipping {task_dir}: no demos.hdf5")
                continue
            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"task.json not found in {task_dir}")

            with open(json_path) as jf:
                instruction = json.load(jf).get("instruction", "")

            with h5py.File(hdf5_path, "r") as f:
                ep_keys = sorted(
                    [k for k in f.keys()
                     if "observations_cam1" in f[k] and "actions" in f[k]],
                    key=lambda k: int(k.split("_")[1]),
                )
                ep_lengths = {k: len(f[k]["actions"]) for k in ep_keys}

            # Episode-level split (not frame-level): chunks within an episode
            # overlap, so splitting by frame would leak near-identical windows
            # into both train and val. Fixed seed independent of the training
            # epoch RNG so the held-out episodes never change across epochs —
            # that's what makes val_loss comparable epoch-to-epoch.
            if val_split > 0 and len(ep_keys) > 1:
                shuffled = list(ep_keys)
                np.random.default_rng(val_seed).shuffle(shuffled)
                n_val         = max(1, round(len(shuffled) * val_split))
                val_ep_keys   = shuffled[:n_val]
                train_ep_keys = shuffled[n_val:]
            else:
                val_ep_keys, train_ep_keys = [], ep_keys

            self._task_meta.append((hdf5_path, instruction, train_ep_keys, ep_lengths))

            for ep_key in val_ep_keys:
                T = ep_lengths[ep_key]
                for t in range(0, T - self.chunk_size, self.frame_stride):
                    self.val_samples.append((hdf5_path, ep_key, t, instruction))

            print(f"[MultiTaskDemoDataset] Scanned {hdf5_path}: "
                  f"{len(ep_keys)} episodes "
                  f"({len(train_ep_keys)} train / {len(val_ep_keys)} val)")

        print(f"[MultiTaskDemoDataset] {len(self._task_meta)} tasks ready, "
              f"{len(self.val_samples)} fixed validation samples. "
              f"Call resample(epoch) to build the training sample index.")

    def resample(self, epoch: int):
        """Randomly select n_episodes_per_task TRAIN episodes per task for this epoch."""
        rng = np.random.default_rng(epoch)
        self.samples = []   # (hdf5_path, ep_key, t, instruction)
        for hdf5_path, instruction, train_ep_keys, ep_lengths in self._task_meta:
            n = min(self.n_episodes_per_task, len(train_ep_keys))
            chosen = rng.choice(train_ep_keys, size=n, replace=False)
            for ep_key in chosen:
                T = ep_lengths[ep_key]
                for t in range(0, T - self.chunk_size, self.frame_stride):
                    self.samples.append((hdf5_path, ep_key, t, instruction))
        print(f"[MultiTaskDemoDataset] Epoch resampled: "
              f"{len(self.samples)} samples "
              f"({self.n_episodes_per_task} eps/task)")

    def _get_handle(self, hdf5_path: str):
        if hdf5_path not in self._handles:
            self._handles[hdf5_path] = h5py.File(hdf5_path, "r")
        return self._handles[hdf5_path]

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, hdf5_path, ep_key, t, instruction):
        f = self._get_handle(hdf5_path)

        def to_tensor(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        cam1 = to_tensor(np.array(f[ep_key]["observations_cam1"][t]))
        cam2 = to_tensor(np.array(f[ep_key]["observations_cam2"][t]))

        chunk = np.array(f[ep_key]["actions"][t : t + self.chunk_size], dtype=np.float32)
        if len(chunk) < self.chunk_size:
            pad   = np.zeros((self.chunk_size - len(chunk), chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        act_chunk = torch.from_numpy(chunk)

        return cam1, cam2, act_chunk, instruction

    def __getitem__(self, idx):
        return self._load_sample(*self.samples[idx])


class MultiTaskValView(TorchDataset):
    """Fixed held-out view over a MultiTaskDemoDataset's val_samples.

    Separate from the main dataset so DataLoader's __len__/__getitem__ see
    only the validation samples, while resample() keeps mutating the parent's
    training self.samples independently.
    """

    def __init__(self, parent: "MultiTaskDemoDataset"):
        self.parent = parent

    def __len__(self):
        return len(self.parent.val_samples)

    def __getitem__(self, idx):
        return self.parent._load_sample(*self.parent.val_samples[idx])


# ========================================================================== #
#  Fine-tuning                                                                #
# ========================================================================== #

def finetune_pi0fast(args):
    """Fine-tune PI0Fast with FSDP across 4 GPUs (torchrun --nproc_per_node=4).

    FSDP shards model weights + optimizer states across all GPUs:
      - Per GPU weight memory: 3.45B × 4 bytes / 4 GPUs ≈ 3.45 GB  (vs 14 GB with DDP)
      - fp32 throughout: no logit overflow, stable cross-entropy on 257k vocab
      - Gradient checkpointing to further reduce activation memory
    """
    _mock_groot()
    import functools
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.utils.data.distributed import DistributedSampler
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    # ── Distributed init ─────────────────────────────────────────────── #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    # ── Resolve resume epoch ─────────────────────────────────────────── #
    start_epoch = 0
    if args.resume_from:
        m = re.match(r".*checkpoint_epoch(\d+)$", args.resume_from.rstrip("/"))
        if not m:
            raise ValueError(
                f"--resume_from must point to checkpoint_epochN, got: {args.resume_from!r}"
            )
        start_epoch = int(m.group(1))
        log(f"\n[finetune] Resuming from: {args.resume_from} (epoch {start_epoch})")
    else:
        log(f"\n[finetune] Base model : {args.model_dir}")

    log(f"[finetune] Output     : {args.finetune_output}")
    log(f"[finetune] Rank {rank}/{world_size}  device={device}")

    # ── Load model on CPU ─────────────────────────────────────────────── #
    # lerobot's PI0FastPolicy.__init__ calls self.model.to(config.device), which
    # defaults to CUDA when GPUs are visible. With 4 ranks each loading 13.8 GB
    # fp32, all go OOM on T4s before FSDP can shard.
    # Fix: temporarily redirect any cuda .to() call to cpu during from_pretrained,
    # then let FSDP move only the local shard (~3.45 GB) to GPU after wrapping.
    model_path = args.model_dir   # always load base weights; LoRA loaded separately below
    log(f"[finetune] Loading PI0FastPolicy from {model_path} (on CPU) …")

    import torch.nn as nn
    _orig_to = nn.Module.to
    def _cpu_only_to(self, *args, **kwargs):
        if args:
            try:
                if torch.device(args[0]).type == "cuda":
                    return _orig_to(self, "cpu")
            except Exception:
                pass
        return _orig_to(self, *args, **kwargs)
    nn.Module.to = _cpu_only_to
    try:
        policy = PI0FastPolicy.from_pretrained(model_path)
    finally:
        nn.Module.to = _orig_to
    policy.config.chunk_size     = args.chunk_size
    policy.config.n_action_steps = args.chunk_size
    policy.train()

    cam1_key = args.cam1_key
    cam2_key = args.cam2_key
    log(f"[finetune] Batch image keys: '{cam1_key}', '{cam2_key}'")

    if args.full_finetune:
        # ── Full fine-tune: every parameter trainable, no LoRA ────────── #
        # Mirrors `lerobot-train --policy.type=pi0` / `pi0_fast` semantics
        # (lerobot's own trainer has no LoRA option — it always fine-tunes
        # the whole model). from_pretrained() already leaves requires_grad
        # True everywhere, but set it explicitly so behavior doesn't depend
        # on that default.
        for p in policy.parameters():
            p.requires_grad_(True)
        log("[finetune] Full fine-tune: all parameters trainable "
            "(no LoRA adapters injected).")

        # ── Load full weights from previous checkpoint (if resuming) ──── #
        if args.resume_from:
            full_pt = os.path.join(args.resume_from, "full_weights.pt")
            if os.path.isfile(full_pt):
                resume_sd = torch.load(full_pt, map_location="cpu")
                missing, unexpected = policy.load_state_dict(resume_sd, strict=False)
                log(f"[finetune] Resumed full model weights from {full_pt} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                log(f"[finetune] WARNING: no full_weights.pt found at {full_pt}")
    else:
        # ── Freeze everything first ───────────────────────────────────── #
        for p in policy.parameters():
            p.requires_grad_(False)

        # ── Apply LoRA (manual, FSDP-compatible) ───────────────────────── #
        # Targets: lm_head + Q/V projections in all transformer layers.
        # Camera views and scene are domain-shifted from pre-training, so the
        # frozen transformer's cross-modal attention needs to adapt — lm_head
        # alone cannot fix poor hidden representations for new visual inputs.
        class LoRALinear(torch.nn.Module):
            def __init__(self, linear, rank, alpha):
                super().__init__()
                in_f, out_f = linear.in_features, linear.out_features
                self.weight = linear.weight
                self.bias   = getattr(linear, "bias", None)
                self.lora_A = torch.nn.Parameter(torch.randn(rank, in_f) * 0.01)
                self.lora_B = torch.nn.Parameter(torch.zeros(out_f, rank))
                self.scale  = alpha / rank

            def forward(self, x):
                out = F.linear(x, self.weight, self.bias)
                out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
                return out

        lora_rank, lora_alpha = args.lora_rank, args.lora_rank * 2

        # Q, K, V, O projections + lm_head.
        pali = policy.model.paligemma_with_expert.paligemma
        for layer in pali.model.language_model.layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, lora_rank, lora_alpha)
            attn.k_proj = LoRALinear(attn.k_proj, lora_rank, lora_alpha)
            attn.v_proj = LoRALinear(attn.v_proj, lora_rank, lora_alpha)
            attn.o_proj = LoRALinear(attn.o_proj, lora_rank, lora_alpha)
        pali.lm_head = LoRALinear(pali.lm_head, lora_rank, lora_alpha)

        # ── Load LoRA weights from previous checkpoint (if resuming) ──── #
        if args.resume_from:
            lora_pt = os.path.join(args.resume_from, "lora_weights.pt")
            if os.path.isfile(lora_pt):
                resume_sd   = torch.load(lora_pt, map_location="cpu")
                lora_params = {n: p for n, p in policy.named_parameters() if "lora_" in n}
                loaded = sum(
                    1 for k, v in resume_sd.items()
                    if k in lora_params and not lora_params[k].data.copy_(v) is None
                )
                log(f"[finetune] Resumed {loaded} LoRA tensors from {lora_pt}")
            else:
                log(f"[finetune] WARNING: no lora_weights.pt found at {lora_pt}")

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    if args.full_finetune:
        log(f"[finetune] Full fine-tune: {n_train:,} / {n_total:,} trainable "
            f"(100%)  chunk={args.chunk_size}")
    else:
        log(f"[finetune] LoRA rank={args.lora_rank} (Q/K/V/O + lm_head): "
            f"{n_train:,} trainable / {n_total:,} total  "
            f"({100*n_train/n_total:.3f}%)  chunk={args.chunk_size}")

    # lerobot's init mixes bf16/fp32 internally; FSDP requires uniform dtype.
    # Use fp32 throughout: no logit overflow, no NaN, at cost of 2× memory vs fp16.
    # FSDP shards across 4 GPUs: 13.8 GB / 4 = 3.45 GB per GPU → fits T4.
    policy = policy.float()

    # ── Gradient checkpointing before FSDP wrapping ───────────────────── #
    if hasattr(policy.model, 'gradient_checkpointing_enable'):
        policy.model.gradient_checkpointing_enable()
        log("[finetune] Gradient checkpointing enabled.")

    # ── FSDP wrap ────────────────────────────────────────────────────── #
    auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000_000)

    # Set the default CUDA device for this rank before FSDP wrapping.
    # FSDP with device_id moves each auto-wrap unit (~100M params, ~400 MB) to
    # GPU one at a time before sharding — never the full 13.8 GB at once.
    # The from_pretrained OOM is prevented by the CPU-load monkey-patch above,
    # so device_id here is safe and correct.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        policy = FSDP(
            policy,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
            use_orig_params=True,
        )
    else:
        policy = policy.to(device)

    log(f"[finetune] FSDP wrapped (fp32). Each GPU holds ~{n_total*4/world_size/1e9:.1f} GB of params.")

    # ── Dataset ──────────────────────────────────────────────────────── #
    if args.tasks:
        task_names = ALL_TASKS if args.tasks == ["all"] else args.tasks
        task_dirs  = [os.path.join(args.data_dir, t) for t in task_names]
        dataset    = MultiTaskDemoDataset(
            task_dirs            = task_dirs,
            chunk_size           = args.chunk_size,
            n_episodes_per_task  = args.episodes_per_task,
            frame_stride         = args.frame_stride,
            val_split            = args.val_split,
        )
        multi_task = True
        val_dataset = MultiTaskValView(dataset) if args.val_split > 0 else None
        log(f"[finetune] Multi-task: {len(task_names)} tasks, "
            f"{args.episodes_per_task} eps/task per epoch"
            + (f", val_split={args.val_split}" if args.val_split > 0 else ""))
    else:
        dataset    = DemoDataset(
            path         = args.demo_data,
            chunk_size   = args.chunk_size,
            frame_stride = args.frame_stride,
        )
        multi_task  = False
        val_dataset = None
        if args.val_split > 0:
            log("[finetune] WARNING: --val_split is only implemented for --tasks "
                "(multi-task) mode; ignoring for this single-task run.")

    # ── Action normalisation stats (per-dim min/max over the FULL dataset) ── #
    # The FAST tokenizer needs actions in ~[-1,1]; raw values overflow it and floor
    # negatives at ~-6.46 (destroying strongly-negative targets). Compute per-dim
    # min/max over EVERY episode (not a per-epoch subset), normalise targets at
    # tokenise time, and persist the stats with the checkpoint so inference
    # denormalises with exactly these numbers.
    action_low, action_high = compute_action_stats(dataset)
    action_dim = len(action_low)
    log(f"[finetune] Action normalisation to [-1,1] (per-dim, full dataset), "
        f"action_dim={action_dim}:")
    for d in range(action_dim):
        log(f"           dim{d}: [{action_low[d]:+.4f}, {action_high[d]:+.4f}]")

    # ── Optimizer and scheduler ──────────────────────────────────────── #
    total_epochs = start_epoch + args.epochs
    trainable    = [p for p in policy.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=args.lr * 0.1,
    )
    if args.resume_from:
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            try:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                log(f"[finetune] Restored optimizer + scheduler from {state_path}")
            except ValueError:
                log(f"[finetune] Optimizer state mismatch (LoRA structure changed) — starting fresh optimizer")

    # ── Shared batch → loss (used by both the training and validation loops) ── #
    def compute_batch_loss(batch_data):
        if multi_task:
            cam1_imgs, cam2_imgs, action_chunks, instructions = batch_data
        else:
            cam1_imgs, cam2_imgs, action_chunks = batch_data
            instructions = None

        cam1_imgs     = cam1_imgs.to(device)
        cam2_imgs     = cam2_imgs.to(device)
        action_chunks = action_chunks.to(device)
        B             = cam1_imgs.size(0)

        # raw_policy for tokenizer access (FSDP wraps forward only)
        raw_policy = policy._fsdp_wrapped_module if world_size > 1 else policy
        act_tokens, act_masks = tokenize_action_chunks(
            action_chunks, raw_policy, action_low, action_high)
        act_tokens = act_tokens.to(device)
        act_masks  = act_masks.to(device)

        if multi_task:
            lang_ids, lang_masks = build_language_tokens_batch(
                list(instructions), args.state_dim, raw_policy, device,
            )
        else:
            lang_ids, lang_masks = build_language_tokens(
                args.instruction, args.state_dim, raw_policy, B, device,
            )

        batch = {
            cam1_key:                              cam1_imgs,
            cam2_key:                              cam2_imgs,
            "observation.language.tokens":         lang_ids,
            "observation.language.attention_mask": lang_masks.bool(),
            "action.tokens":                       act_tokens,
            "action.token_mask":                   act_masks.bool(),
        }
        loss, loss_dict = policy(batch)
        return loss, loss_dict, B

    # ── Training loop ────────────────────────────────────────────────── #
    os.makedirs(args.finetune_output, exist_ok=True)
    nan_skip_count = 0

    for ep in range(start_epoch, total_epochs):
        if multi_task:
            dataset.resample(ep)

        if world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=ep,
            )
            sampler.set_epoch(ep)
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=sampler, num_workers=0, drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0, drop_last=True)

        policy.train()
        total_loss = 0.0
        n_steps    = 0

        step_bar = tqdm(loader, desc=f"  Epoch {ep+1}/{total_epochs} [r{rank}]",
                        unit="batch", leave=False, disable=(rank != 0))

        for batch_data in step_bar:
            optimizer.zero_grad()
            loss, loss_dict, B = compute_batch_loss(batch_data)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_skip_count += 1
                if rank == 0:
                    print(f"[finetune] WARNING: NaN/Inf loss (total skipped={nan_skip_count}), skipping batch",
                          flush=True)
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps    += 1
            if rank == 0:
                step_bar.set_postfix(loss=f"{loss.item():.4f}",
                                     ce=f"{loss_dict.get('ce_loss', 0):.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        scheduler.step()

        # ── Validation: fixed held-out episodes, forward-only ──────────── #
        val_loss_str = ""
        if val_dataset is not None and len(val_dataset) > 0:
            policy.eval()
            if world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset, num_replicas=world_size, rank=rank,
                    shuffle=False, drop_last=False,
                )
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                        sampler=val_sampler, num_workers=0)
            else:
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=0)

            val_loss_sum = torch.zeros(1, device=device)
            val_count    = torch.zeros(1, device=device)
            with torch.no_grad():
                for batch_data in val_loader:
                    v_loss, _, v_B = compute_batch_loss(batch_data)
                    if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                        val_loss_sum += v_loss.item() * v_B
                        val_count    += v_B

            if world_size > 1:
                dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_count,    op=dist.ReduceOp.SUM)
            avg_val_loss = (val_loss_sum / val_count.clamp(min=1)).item()
            val_loss_str = f"  val_loss={avg_val_loss:.4f}"
            policy.train()

        log(f"[finetune] Epoch {ep+1}/{total_epochs}  "
            f"loss={avg_loss:.4f}{val_loss_str}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"nan_skipped={nan_skip_count}")

        # ── Checkpoint: save LoRA weights only (rank 0) ──────────────── #
        if world_size > 1:
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, save_cfg):
                full_sd = policy.state_dict()
        else:
            full_sd = policy.state_dict()

        if rank == 0:
            ckpt_dir = os.path.join(args.finetune_output, f"checkpoint_epoch{ep+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            if args.full_finetune:
                torch.save(full_sd, os.path.join(ckpt_dir, "full_weights.pt"))
                log(f"[finetune] Checkpoint → {ckpt_dir}  ({len(full_sd)} tensors, full model)")
            else:
                lora_sd = {k: v for k, v in full_sd.items() if "lora_" in k}
                torch.save(lora_sd, os.path.join(ckpt_dir, "lora_weights.pt"))
                log(f"[finetune] Checkpoint → {ckpt_dir}  ({len(lora_sd)} LoRA tensors, incl. lm_head)")
            torch.save(
                {"optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "epoch":     ep + 1},
                os.path.join(ckpt_dir, "training_state.pt"),
            )
            # Persist the normalisation stats WITH the checkpoint (inference reads
            # them from here to denormalise — must match what tokenisation used).
            save_norm_stats(ckpt_dir, action_low, action_high)
            if ep > start_epoch:
                prev = os.path.join(args.finetune_output, f"checkpoint_epoch{ep}")
                if os.path.isdir(prev):
                    shutil.rmtree(prev)

        if world_size > 1:
            dist.barrier()

    # ── Final save ───────────────────────────────────────────────────── #
    if world_size > 1:
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, save_cfg):
            full_sd = policy.state_dict()
    else:
        full_sd = policy.state_dict()

    if rank == 0:
        if args.full_finetune:
            torch.save(full_sd, os.path.join(args.finetune_output, "full_weights.pt"))
            log(f"\n[finetune] Done. Full model weights → {args.finetune_output}/full_weights.pt")
        else:
            lora_sd = {k: v for k, v in full_sd.items() if "lora_" in k}
            torch.save(lora_sd, os.path.join(args.finetune_output, "lora_weights.pt"))
            log(f"\n[finetune] Done. LoRA weights → {args.finetune_output}/lora_weights.pt")
            log(f"[finetune] Load: replace lm_head with LoRALinear, then load lora_weights.pt")
        save_norm_stats(args.finetune_output, action_low, action_high)
        log(f"[finetune] Normalisation stats → {args.finetune_output}/norm_stats.json")

    if world_size > 1:
        dist.destroy_process_group()


# ========================================================================== #
#  CLI                                                                        #
# ========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="PI0Fast fine-tuning on dual-camera Franka demonstrations"
    )

    # Model
    p.add_argument("--model_dir", type=str,
                   default="physical-intelligence/pi0-fast",
                   help="HF hub ID or local path to base PI0Fast weights")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint_epochN directory to resume from")
    p.add_argument("--finetune_output", type=str,
                   default="outputs/pi0fast_finetuned")

    # Camera keys — override auto-detected keys from model config
    p.add_argument("--cam1_key", type=str, default="observation.images.base_0_rgb",
                   help="Batch key for camera 1 (observations_cam1)")
    p.add_argument("--cam2_key", type=str, default="observation.images.right_wrist_0_rgb",
                   help="Batch key for camera 2 (observations_cam2)")

    # Data — multi-task (--tasks) or single-task (--demo_data)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Task name(s) or 'all'. Reads data/<task>/demos.hdf5 + task.json. "
                        "Takes precedence over --demo_data when provided.")
    p.add_argument("--data_dir", type=str, default="data",
                   help="Root directory containing per-task data folders (used with --tasks).")
    p.add_argument("--demo_data", type=str,
                   default="data/pick_blue_cube_to_tray/demos.hdf5",
                   help="Single HDF5 file path (single-task mode; ignored when --tasks is set).")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Sample every Nth frame per episode")
    p.add_argument("--episodes_per_task", type=int, default=25,
                   help="Episodes to sample per task per epoch (lazy resampling)")
    p.add_argument("--val_split", type=float, default=0.0,
                   help="Fraction of episodes per task held out for validation "
                        "(episode-level split — whole episodes, not frames, so "
                        "overlapping chunks never leak across the split). "
                        "Multi-task (--tasks) mode only. 0 = no split (default). "
                        "When set, reports val_loss each epoch from a forward-only "
                        "pass over the fixed held-out episodes.")

    # Training
    p.add_argument("--chunk_size",  type=int,   default=10,
                   help="Action chunk length (default 10 matches Stage 1 num_pred)")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--lr",          type=float, default=2.5e-5)
    p.add_argument("--state_dim",   type=int,   default=6,
                   help="Robot state dim for language prompt (zeros used)")
    p.add_argument("--instruction", type=str,
                   default="pick up the blue cube and place it on the tray")

    # LoRA / full fine-tune
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank for lm_head fine-tuning (default 16 ≈ 8M params). "
                        "Ignored when --full_finetune is set.")
    p.add_argument("--full_finetune", action="store_true",
                   help="Train every parameter instead of injecting LoRA adapters — "
                        "mirrors `lerobot-train --policy.type=pi0` / `pi0_fast` semantics. "
                        "Needs far more VRAM/host RAM: fp32 AdamW keeps ~3x the model size "
                        "(params + 2 optimizer moments) sharded across GPUs, vs ~0.2% of "
                        "that for LoRA. Checkpoints save full_weights.pt (~13.8GB) instead "
                        "of lora_weights.pt.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune_pi0fast(args)
