"""
openvla_oft_server.py  —  OpenVLA-OFT fine-tuning and inference on Franka demos.

Mirrors the structure of pi0fast_server.py but targets OpenVLA-7B with the
OFT (Often Fine-Tuned) parallel action decoding head instead of PI0Fast.

OpenVLA-OFT replaces autoregressive action token decoding with a lightweight
MLP head that predicts all action dimensions in a single forward pass:
  image + instruction  →  VLM hidden state  →  action chunk (chunk_size × 7)

Architecture
------------
  Base model  : openvla/openvla-7b  (SigLIP-400M + DINOv2-L + Llama-2-7B)
  LoRA        : rank-32 adapters on Q/K/V/O projections in all LLM layers
  Action head : LayerNorm + Linear(4096 → 2048) + GELU + Linear(2048 → action_dim × chunk_size) + Tanh
  Loss        : L2 (MSE) on continuous actions  (no action discretisation)

HDF5 layout (data/<task>/demos.hdf5):
    /episode_N/observations_cam1  (T, 224, 224, 3)  uint8
    /episode_N/observations_cam2  (T, 224, 224, 3)  uint8  [optional; falls back to cam1]
    /episode_N/actions            (T, action_dim)   float32

Usage
-----
Multi-GPU fine-tune (all tasks):
    torchrun --nproc_per_node=4 server/openvla_oft_server.py \\
        --model_dir openvla/openvla-7b \\
        --tasks all \\
        --finetune_output outputs/openvla_oft_finetuned \\
        --epochs 20 --batch_size 2

Single-task fine-tune:
    python server/openvla_oft_server.py \\
        --model_dir openvla/openvla-7b \\
        --demo_data data/pick_red_cube_to_tray/demos.hdf5 \\
        --finetune_output outputs/openvla_oft_finetuned \\
        --instruction "pick up the red cube and place it on the tray"

Resume from checkpoint:
    torchrun --nproc_per_node=4 server/openvla_oft_server.py \\
        --resume_from outputs/openvla_oft_finetuned/checkpoint_epoch5 \\
        --tasks all --finetune_output outputs/openvla_oft_finetuned

Inference server (TCP, same protocol as pi0fast_server.py):
    python server/openvla_oft_server.py --serve \\
        --model_dir openvla/openvla-7b \\
        --lora_path outputs/openvla_oft_finetuned/lora_weights.pt \\
        --head_path outputs/openvla_oft_finetuned/action_head.pt
"""

import argparse
import base64
import io
import json
import os
import random
import re
import shutil
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm


# ========================================================================== #
#  Task list (mirrors pi0fast_server.py)                                      #
# ========================================================================== #

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


# ========================================================================== #
#  LoRA linear layer                                                           #
# ========================================================================== #

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation.

    param_dtype controls lora_A/lora_B storage dtype (default: same as frozen weight).
    Set param_dtype=torch.float32 for vision LoRA to keep gradients in fp32 and
    prevent fp16 overflow on long backward paths through the vision backbone.
    The forward cast (fp32 → x.dtype) adds negligible cost.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: int,
                 param_dtype: torch.dtype = None):
        super().__init__()
        dev   = linear.weight.device
        dtype = linear.weight.dtype
        in_f, out_f = linear.in_features, linear.out_features
        self.weight = linear.weight
        self.bias   = getattr(linear, "bias", None)
        lora_dtype  = param_dtype if param_dtype is not None else dtype
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=dev, dtype=lora_dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=lora_dtype))
        self.scale  = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        # Cast LoRA params to computation dtype if they differ (e.g. fp32 params, fp16 input).
        # Casting the PARAMETER (not the input x) is safe under FSDP.
        lA = self.lora_A if self.lora_A.dtype == x.dtype else self.lora_A.to(x.dtype)
        lB = self.lora_B if self.lora_B.dtype == x.dtype else self.lora_B.to(x.dtype)
        out = out + F.linear(F.linear(x, lA), lB) * self.scale
        return out


def insert_lora(model: nn.Module, target_suffixes=("q_proj", "k_proj", "v_proj", "o_proj"),
                rank: int = 32, alpha: int = 64,
                param_dtype: torch.dtype = None) -> int:
    """
    Replace all nn.Linear layers whose name ends with a target suffix with
    LoRALinear in-place.  Returns the number of layers replaced.
    """
    replaced = 0
    for full_name, module in list(model.named_modules()):
        for suffix in target_suffixes:
            if full_name.endswith(f".{suffix}") and isinstance(module, nn.Linear):
                parent_name, attr = full_name.rsplit(".", 1)
                parent = model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                setattr(parent, attr, LoRALinear(module, rank, alpha, param_dtype=param_dtype))
                replaced += 1
                break
    return replaced


# ========================================================================== #
#  FiLM (Feature-wise Linear Modulation) for vision-language conditioning     #
# ========================================================================== #

class FiLMGenerator(nn.Module):
    """
    Maps a mean-pooled instruction embedding → per-block (gamma, beta) pairs.
    Kept OUTSIDE the FSDP-wrapped model (like ActionChunkHead) so it can be
    DDP-wrapped and stay in fp32.  Zero-initialized heads → identity at init.
    """
    def __init__(self, lang_dim: int, film_hidden: int, feat_dims: list):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, film_hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(film_hidden, d * 2) for d in feat_dims
        ])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, lang_emb: torch.Tensor) -> list:
        # lang_emb: (B, lang_dim) float32
        h = self.shared(lang_emb.float())
        return [head(h).chunk(2, dim=-1) for head in self.heads]  # [(gamma,beta), ...]


class FiLMBlock(nn.Module):
    """
    Wraps a vision transformer block; applies FiLM after it.
    Reads pre-computed [(gamma, beta), ...] from lang_holder[0].
    lang_holder[0] is None when FiLM conditioning is disabled.
    """
    def __init__(self, block: nn.Module, block_idx: int, lang_holder: list):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.lang_holder = lang_holder

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.block(x, **kwargs)
        film_params = self.lang_holder[0]
        if film_params is not None:
            gamma, beta = film_params[self.block_idx]
            gamma = gamma.unsqueeze(1).to(x.dtype)   # (B, 1, feat_dim)
            beta  = beta.unsqueeze(1).to(x.dtype)
            x = (1.0 + gamma) * x + beta             # residual: identity when zero-init
        return x


def insert_film(model: nn.Module, lang_dim: int,
                film_hidden: int = 256, n_last_blocks: int = 4):
    """
    Wrap the last n_last_blocks of each vision encoder block-list with FiLMBlock.
    Returns (film_gen, lang_holder).  film_gen is None if no vision backbone found.
    """
    lang_holder = [None]
    vb = getattr(model, "vision_backbone", None)
    if vb is None:
        return None, lang_holder

    target_blocks = []
    feat_dims = []
    for enc_name in ("featurizer", "fused_featurizer"):
        enc = getattr(vb, enc_name, None)
        if enc is None:
            continue
        blocks = getattr(enc, "blocks", None)
        if blocks is None:
            continue
        n = len(blocks)
        for i in range(max(0, n - n_last_blocks), n):
            blk = blocks[i]
            target_blocks.append((blocks, i))
            norm = getattr(blk, "norm1", None) or getattr(blk, "norm", None)
            feat_dim = norm.weight.shape[0] if (norm is not None and hasattr(norm, "weight")) else 1152
            feat_dims.append(feat_dim)

    if not target_blocks:
        return None, lang_holder

    film_gen = FiLMGenerator(lang_dim, film_hidden, feat_dims)
    for idx, (blocks_list, i) in enumerate(target_blocks):
        blocks_list[i] = FiLMBlock(blocks_list[i], idx, lang_holder)

    return film_gen, lang_holder


# ========================================================================== #
#  OFT action chunk head                                                       #
# ========================================================================== #

class ActionChunkHead(nn.Module):
    """
    Maps the LLM's last-token hidden state to a chunk of continuous actions.

    Input  : (B, hidden_size)  — hidden state at the final input token ("Out:")
    Output : (B, chunk_size, action_dim)  in [-1, 1]
    """

    def __init__(self, hidden_size: int = 4096, action_dim: int = 7, chunk_size: int = 16):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        mid = hidden_size // 2
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Linear(mid, action_dim * chunk_size),
            nn.Tanh(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, hidden_size)
        out = self.net(hidden)                                   # (B, action_dim * chunk_size)
        return out.view(-1, self.chunk_size, self.action_dim)    # (B, K, D)


# ========================================================================== #
#  HDF5 datasets  (same format as pi0fast_server.py)                          #
# ========================================================================== #

class DemoDataset(TorchDataset):
    """
    Single-task dataset.  Returns (cam1_img, act_chunk, instruction).

    cam1: (3, 224, 224) float [0, 1]
    act_chunk: (chunk_size, action_dim) float32
    """

    def __init__(
        self,
        path:         str,
        instruction:  str,
        chunk_size:   int = 16,
        frame_stride: int = 1,
    ):
        self.chunk_size  = chunk_size
        self.instruction = instruction
        self._cam1:  list = []
        self._acts:  list = []
        self.samples: list = []

        print(f"[DemoDataset] Loading {path} …")
        with h5py.File(path, "r") as f:
            ep_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
            for ep_key in ep_keys:
                grp = f[ep_key]
                if "observations_cam1" not in grp or "actions" not in grp:
                    continue
                self._cam1.append(np.array(grp["observations_cam1"]))
                self._acts.append(np.array(grp["actions"], dtype=np.float32))

        for ep_idx, acts in enumerate(self._acts):
            T = len(acts)
            for t in range(0, T - chunk_size, frame_stride):
                self.samples.append((ep_idx, t))

        print(f"[DemoDataset] {len(self._acts)} episodes → {len(self.samples)} samples "
              f"(chunk={chunk_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]

        def to_tensor(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        cam1 = to_tensor(self._cam1[ep_idx][t])

        chunk = self._acts[ep_idx][t : t + self.chunk_size]
        if len(chunk) < self.chunk_size:
            pad   = np.zeros((self.chunk_size - len(chunk), chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        act_chunk = torch.from_numpy(chunk)

        return cam1, act_chunk, self.instruction


class MultiTaskDemoDataset(TorchDataset):
    """
    Lazy-loading multi-task dataset.  Metadata-only at init; reads frames
    on-demand from HDF5.  Call resample(epoch) to pick fresh episodes each epoch.
    """

    def __init__(
        self,
        task_dirs:            list,
        chunk_size:           int   = 16,
        n_episodes_per_task:  int   = 25,
        frame_stride:         int   = 1,
        non_transition_prob:  float = 0.0,
        transition_lookahead: int   = 16,
    ):
        self.chunk_size           = chunk_size
        self.n_episodes_per_task  = n_episodes_per_task
        self.frame_stride         = frame_stride
        self.non_transition_prob  = non_transition_prob
        self.transition_lookahead = transition_lookahead

        self._task_meta = []   # (hdf5_path, instruction, ep_keys, ep_lengths)
        self._handles   = {}
        self.samples    = []

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
            self._task_meta.append((hdf5_path, instruction, ep_keys, ep_lengths))
            print(f"[MultiTaskDemoDataset] Scanned {hdf5_path}: {len(ep_keys)} episodes")

        print(f"[MultiTaskDemoDataset] {len(self._task_meta)} tasks ready. "
              f"Call resample(epoch) to build sample index.")

    def resample(self, epoch: int):
        rng = np.random.default_rng(epoch)
        self.samples = []
        for hdf5_path, instruction, ep_keys, ep_lengths in self._task_meta:
            n      = min(self.n_episodes_per_task, len(ep_keys))
            chosen = rng.choice(ep_keys, size=n, replace=False)
            for ep_key in chosen:
                T = ep_lengths[ep_key]
                for t in range(0, T - self.chunk_size, self.frame_stride):
                    self.samples.append((hdf5_path, ep_key, t, instruction))

        # True skipping: keep all transition chunks; keep non-transition with prob=non_transition_prob
        # non_transition_prob=0.0 → keep all non-transition (off)
        # non_transition_prob=0.1 → keep 10% of non-transition
        # non_transition_prob=1.0 → keep all non-transition (same as off)
        if 0.0 < self.non_transition_prob < 1.0:
            keep = []
            n_transition = 0
            rand_vals = rng.random(len(self.samples))
            for i, (hdf5_path, ep_key, t, _) in enumerate(self.samples):
                f       = self._get_handle(hdf5_path)
                gripper = np.array(f[ep_key]["actions"][t : t + self.transition_lookahead, 6])
                is_transition = (np.abs(np.diff(gripper)) > 0.5).any()
                if is_transition:
                    keep.append(self.samples[i])
                    n_transition += 1
                elif rand_vals[i] < self.non_transition_prob:
                    keep.append(self.samples[i])
            self.samples = keep
            print(f"[MultiTaskDemoDataset] non_transition_prob={self.non_transition_prob:.2f}: "
                  f"{n_transition} transition / {len(self.samples)} total kept")

        print(f"[MultiTaskDemoDataset] Epoch resampled: {len(self.samples)} samples")

    def _get_handle(self, hdf5_path):
        if hdf5_path not in self._handles:
            self._handles[hdf5_path] = h5py.File(hdf5_path, "r")
        return self._handles[hdf5_path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, ep_key, t, instruction = self.samples[idx]
        f = self._get_handle(hdf5_path)

        def to_tensor(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        cam1 = to_tensor(np.array(f[ep_key]["observations_cam1"][t]))

        chunk = np.array(f[ep_key]["actions"][t : t + self.chunk_size], dtype=np.float32)
        if len(chunk) < self.chunk_size:
            pad   = np.zeros((self.chunk_size - len(chunk), chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        act_chunk = torch.from_numpy(chunk)

        gripper_ahead = np.array(f[ep_key]["actions"][t : t + self.transition_lookahead, 6])
        has_near_transition = float((np.abs(np.diff(gripper_ahead)) > 0.5).any()) if len(gripper_ahead) > 1 else 0.0

        return cam1, act_chunk, instruction, has_near_transition


# ========================================================================== #
#  Image + prompt helpers                                                      #
# ========================================================================== #

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) float [0,1] → PIL RGB."""
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_prompt(instruction: str) -> str:
    """OpenVLA prompt format."""
    return f"In: What action should the robot take to {instruction}?\nOut:"


# ========================================================================== #
#  Fine-tuning                                                                 #
# ========================================================================== #

def finetune_openvla_oft(args):
    """
    Fine-tune OpenVLA-7B with an OFT action chunk head.

    Uses FSDP across N GPUs (torchrun --nproc_per_node=N) for memory efficiency:
      - Base model (~14 GB fp32) sharded across GPUs: ~3.5 GB per GPU on 4×
      - Only LoRA matrices and action head have gradients
      - Gradient checkpointing further reduces activation memory
    """
    import functools
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoModelForVision2Seq, AutoProcessor

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
                f"--resume_from must end with checkpoint_epochN, got: {args.resume_from!r}"
            )
        start_epoch = int(m.group(1))
        log(f"\n[finetune] Resuming from: {args.resume_from} (epoch {start_epoch})")
    else:
        log(f"\n[finetune] Base model : {args.model_dir}")

    log(f"[finetune] Output     : {args.finetune_output}")
    log(f"[finetune] Rank {rank}/{world_size}  device={device}")

    # ── Load processor (all ranks, lightweight) ──────────────────────── #
    log(f"[finetune] Loading processor from {args.model_dir} …")
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, use_fast=True,
    )

    # ── Load model on CPU to avoid OOM before FSDP sharding ─────────── #
    # Temporarily redirect .to(cuda) → .to(cpu) during from_pretrained,
    # then let FSDP move only the local shard to GPU after wrapping.
    log(f"[finetune] Loading AutoModelForVision2Seq from {args.model_dir} (on CPU) …")
    _orig_to = nn.Module.to

    def _cpu_only_to(self, *a, **kw):
        if a:
            try:
                if torch.device(a[0]).type == "cuda":
                    return _orig_to(self, "cpu")
            except Exception:
                pass
        return _orig_to(self, *a, **kw)

    nn.Module.to = _cpu_only_to
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    finally:
        nn.Module.to = _orig_to

    model.train()

    # ── Freeze all parameters ────────────────────────────────────────── #
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Insert LoRA into LLM attention layers only ───────────────────── #
    n_llm = 0
    if args.lora_rank > 0:
        lora_alpha = args.lora_rank * 2
        llm_module = getattr(model, "language_model", model)
        n_llm = insert_lora(
            llm_module,
            target_suffixes=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=args.lora_rank,
            alpha=lora_alpha,
        )
        log(f"[finetune] Inserted LoRA into {n_llm} LLM layers "
            f"(rank={args.lora_rank}, alpha={lora_alpha})")
    else:
        log("[finetune] LLM LoRA disabled (--lora_rank 0) — LLM fully frozen")

    # ── Insert LoRA into last few blocks of both vision encoders ─────── #
    vis_alpha = args.lora_rank_vision * 2
    vb = getattr(model, "vision_backbone", None)
    n_vis = 0
    if vb is not None and args.lora_rank_vision > 0:
        n_last = args.vision_lora_last_blocks
        for enc_name in ("featurizer", "fused_featurizer"):
            enc = getattr(vb, enc_name, None)
            if enc is None:
                continue
            blocks = getattr(enc, "blocks", None)
            if blocks is None:
                continue
            n_blocks = len(blocks)
            for i in range(max(0, n_blocks - n_last), n_blocks):
                n_vis += insert_lora(
                    blocks[i],
                    target_suffixes=("qkv", "proj"),
                    rank=args.lora_rank_vision,
                    alpha=vis_alpha,
                )
        log(f"[finetune] Inserted LoRA into {n_vis} vision layers "
            f"(rank={args.lora_rank_vision}, alpha={vis_alpha}, "
            f"last {n_last} blocks of each encoder)")

    # ── Detect LLM hidden size ───────────────────────────────────────── #
    # Works for Llama-2 (hidden_size=4096) and Mistral (hidden_size=4096).
    hidden_size = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" in name:
            hidden_size = module.in_features
            break
    if hidden_size is None:
        # Fallback: check config
        hidden_size = getattr(
            getattr(model, "config", None), "text_config", None
        )
        hidden_size = getattr(hidden_size, "hidden_size", None) if hidden_size else None
        hidden_size = hidden_size or 4096
    log(f"[finetune] LLM hidden_size={hidden_size}")

    # ── Load LoRA weights from previous checkpoint ───────────────────── #
    if args.resume_from:
        lora_pt = os.path.join(args.resume_from, "lora_weights.pt")
        if os.path.isfile(lora_pt):
            resume_sd   = torch.load(lora_pt, map_location="cpu")
            lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
            loaded = sum(
                1 for k, v in resume_sd.items()
                if k in lora_params and not lora_params[k].data.copy_(v) is None
            )
            log(f"[finetune] Resumed {loaded} LoRA tensors from {lora_pt}")
        else:
            log(f"[finetune] WARNING: no lora_weights.pt at {lora_pt}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"[finetune] LoRA trainable: {n_train:,} / {n_total:,} "
        f"({100*n_train/n_total:.3f}%)")

    # ── FiLM conditioning in vision encoder (optional) ───────────────── #
    film_gen    = None
    lang_holder = [None]
    embed_weight = None
    if args.use_film:
        film_gen, lang_holder = insert_film(
            model, lang_dim=hidden_size,
            film_hidden=args.film_hidden,
            n_last_blocks=args.vision_lora_last_blocks,
        )
        if film_gen is not None:
            n_film = sum(p.numel() for p in film_gen.parameters())
            log(f"[finetune] FiLM inserted: {len(film_gen.heads)} blocks, "
                f"{n_film:,} params (film_hidden={args.film_hidden})")
            embed_weight = model.language_model.model.embed_tokens.weight.detach().clone()

    # ── Action chunk head (OFT) ──────────────────────────────────────── #
    action_head = ActionChunkHead(
        hidden_size = hidden_size,
        action_dim  = args.action_dim,
        chunk_size  = args.chunk_size,
    )
    if args.resume_from:
        head_pt = os.path.join(args.resume_from, "action_head.pt")
        if os.path.isfile(head_pt):
            action_head.load_state_dict(torch.load(head_pt, map_location="cpu"))
            log(f"[finetune] Action head resumed from {head_pt}")

    log(f"[finetune] ActionChunkHead: "
        f"{sum(p.numel() for p in action_head.parameters()):,} params  "
        f"chunk={args.chunk_size}  action_dim={args.action_dim}")

    # ── fp16 for model (halves memory vs fp32), fp32 for action head ─────── #
    # fp16: 7.5B params × 2 bytes = ~15 GB, sharded across 4 GPUs = ~3.75 GB/GPU
    model       = model.half()
    action_head = action_head.float()   # head stays fp32 for stable gradients

    # ── Gradient checkpointing (LLM + vision, before FSDP) ──────────── #
    llm = getattr(model, "language_model", None)
    if llm is not None and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
        log("[finetune] Gradient checkpointing enabled (LLM).")
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log("[finetune] Gradient checkpointing enabled (full model).")

    # Vision gradient checkpointing is intentionally NOT enabled:
    # with batch_size=1, the 51-block vision activation memory is ~25 MB
    # (negligible), and TIMM's set_grad_checkpointing interacts poorly with
    # FSDP's use_orig_params=True, causing NaN gradients on the first backward.

    # ── FSDP wrap ────────────────────────────────────────────────────── #
    # Use transformer_auto_wrap_policy targeting only LlamaDecoderLayer so the
    # TIMM vision backbone's ModuleList (blocks) is NOT individually FSDP-wrapped.
    # TIMM's get_intermediate_layers calls len(self.blocks) during forward, which
    # fails when blocks is an FSDP wrapper (no __len__).
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        _wrap_cls = {LlamaDecoderLayer}
    except ImportError:
        _wrap_cls = set()
        log("[finetune] WARNING: LlamaDecoderLayer not found, using no inner wrap policy")

    auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=_wrap_cls) \
                if _wrap_cls else None

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
            use_orig_params=True,
        )
        action_head = action_head.to(device)
        from torch.nn.parallel import DistributedDataParallel as DDP
        action_head = DDP(action_head, device_ids=[local_rank])
        if film_gen is not None:
            film_gen = film_gen.float().to(device)
            film_gen = DDP(film_gen, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model       = model.to(device)
        action_head = action_head.to(device)
        if film_gen is not None:
            film_gen = film_gen.float().to(device)

    if embed_weight is not None:
        embed_weight = embed_weight.to(device)

    # Resume FiLM weights after DDP wrapping
    if args.use_film and film_gen is not None and args.resume_from:
        film_pt = os.path.join(args.resume_from, "film_weights.pt")
        if os.path.isfile(film_pt):
            raw_film = film_gen.module if hasattr(film_gen, "module") else film_gen
            raw_film.load_state_dict(torch.load(film_pt, map_location="cpu"))
            log(f"[finetune] FiLM weights resumed from {film_pt}")

    log(f"[finetune] Model wrapped.  Each GPU ≈ "
        f"{n_total*2/world_size/1e9:.1f} GB of fp16 params.")

    # ── Dataset ──────────────────────────────────────────────────────── #
    total_epochs = start_epoch + args.epochs

    if args.tasks:
        task_names = ALL_TASKS if args.tasks == ["all"] else args.tasks
        task_dirs  = [os.path.join(args.data_dir, t) for t in task_names]
        dataset    = MultiTaskDemoDataset(
            task_dirs             = task_dirs,
            chunk_size            = args.chunk_size,
            n_episodes_per_task   = args.episodes_per_task,
            frame_stride          = args.frame_stride,
            non_transition_prob   = args.non_transition_prob,
            transition_lookahead  = args.transition_lookahead,
        )
        multi_task = True
        log(f"[finetune] Multi-task: {len(task_names)} tasks, "
            f"{args.episodes_per_task} eps/task per epoch")
    else:
        dataset = DemoDataset(
            path         = args.demo_data,
            instruction  = args.instruction,
            chunk_size   = args.chunk_size,
            frame_stride = args.frame_stride,
        )
        multi_task = False

    # ── Optimizer ────────────────────────────────────────────────────── #
    lora_params  = [p for p in model.parameters() if p.requires_grad]
    head_params  = list(action_head.parameters())
    film_params  = list(film_gen.parameters()) if film_gen is not None else []
    param_groups = [
        {"params": lora_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 5.0},
    ]
    if film_params:
        param_groups.append({"params": film_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.95), eps=1e-4, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=args.lr * 0.1,
    )
    if args.resume_from and not args.reset_optimizer:
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            try:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                log(f"[finetune] Restored optimizer + scheduler from {state_path}")
            except (ValueError, KeyError):
                log("[finetune] Optimizer state mismatch — starting fresh optimizer")
    elif args.reset_optimizer:
        log(f"[finetune] --reset_optimizer: starting fresh optimizer/scheduler at lr={args.lr:.2e}")

    # ── Gradient scaler for fp16 FSDP ────────────────────────────────── #
    try:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
        log("[finetune] Using ShardedGradScaler for fp16 gradient scaling")
    except ImportError:
        scaler = torch.cuda.amp.GradScaler()
        log("[finetune] Using GradScaler (ShardedGradScaler not available)")

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

        model.train()
        action_head.train()
        total_loss = 0.0
        n_steps    = 0

        step_bar = tqdm(loader, desc=f"  Epoch {ep+1}/{total_epochs} [r{rank}]",
                        unit="batch", leave=False, disable=(rank != 0))

        for cam1_imgs, action_chunks, instructions, near_transition in step_bar:
            B             = cam1_imgs.size(0)
            action_chunks = action_chunks.to(device)
            near_transition = near_transition.float().to(device)   # (B,)

            # Build prompt text per sample
            prompts = [build_prompt(instr) for instr in instructions]

            # Process images + text through the OpenVLA processor
            pil_cam1 = [tensor_to_pil(cam1_imgs[i]) for i in range(B)]

            try:
                inputs = processor(
                    text   = prompts,
                    images = pil_cam1,
                    return_tensors = "pt",
                    padding        = True,
                    truncation     = True,
                ).to(device)
            except Exception as e:
                log(f"[finetune] WARNING: processor error ({e}), skipping batch")
                continue

            # Forward pass: get hidden states at last prefix token
            optimizer.zero_grad()
            try:
                # Cast pixel_values to fp16 to match model weights
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].half()

                # Compute FiLM language conditioning before vision forward
                if film_gen is not None and embed_weight is not None:
                    text_enc = processor.tokenizer(
                        list(instructions), return_tensors="pt",
                        padding=True, truncation=True,
                    ).to(device)
                    with torch.no_grad():
                        text_emb = F.embedding(text_enc.input_ids, embed_weight.half())
                    mask = text_enc.attention_mask.unsqueeze(-1).float()
                    lang_emb = (text_emb.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                    lang_holder[0] = film_gen(lang_emb)  # [(gamma,beta), ...]

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                lang_holder[0] = None  # clear after forward
                # last_hidden: (B, seq_len, hidden_size)
                last_hidden = outputs.hidden_states[-1]
                # Take the final input token's representation (the ":" of "Out:")
                prefix_feat = last_hidden[:, -1, :]     # (B, hidden_size)
            except Exception as e:
                import traceback as _tb
                lang_holder[0] = None
                log(f"[finetune] WARNING: forward error:\n{_tb.format_exc()}")
                optimizer.zero_grad()
                continue

            pred_actions = action_head(prefix_feat.float())   # (B, chunk_size, action_dim)
            xy_loss      = F.mse_loss(pred_actions[..., :2], action_chunks[..., :2])
            z_loss       = F.mse_loss(pred_actions[..., 2],  action_chunks[..., 2])

            # Upweight chunks near a gripper transition (within transition_lookahead steps)
            chunk_weights = 1.0 + 9.0 * near_transition   # (B,)

            step_losses  = ((pred_actions[..., :3] - action_chunks[..., :3])**2).mean(dim=-1)  # (B, H)
            pos_loss     = (chunk_weights * step_losses.mean(dim=-1)).mean()
            pred_gripper = ((pred_actions[..., 6] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            tgt_gripper  = (action_chunks[..., 6] + 1) / 2
            gripper_loss = F.binary_cross_entropy(pred_gripper, tgt_gripper)
            loss = pos_loss + 0.1 * gripper_loss

            if rank == 0 and n_steps < 10:
                n_trans = near_transition.sum().item()
                print(f"[loss_debug] step={n_steps}  xy={xy_loss.item():.4f}  z={z_loss.item():.4f}  "
                      f"pos={pos_loss.item():.4f}  bce={gripper_loss.item():.4f}  "
                      f"n_near_transition={n_trans:.0f}  total={loss.item():.4f}", flush=True)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_skip_count += 1
                if rank == 0:
                    print(f"[finetune] WARNING: NaN/Inf loss (total skipped={nan_skip_count})",
                          flush=True)
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            all_params = lora_params + head_params + film_params
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                nan_skip_count += 1
                if rank == 0:
                    print(f"[finetune] WARNING: non-finite grad norm={grad_norm:.4f} "
                          f"(total skipped={nan_skip_count})", flush=True)
                optimizer.zero_grad()
            scaler.update()

            total_loss += loss.item()
            n_steps    += 1
            if rank == 0:
                step_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        scheduler.step()
        log(f"[finetune] Epoch {ep+1}/{total_epochs}  "
            f"loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"nan_skipped={nan_skip_count}")

        # ── Checkpoint ───────────────────────────────────────────────── #
        if world_size > 1:
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
                full_sd = model.state_dict()
            raw_head = action_head.module if hasattr(action_head, "module") else action_head
            head_sd  = raw_head.state_dict()
            # All-reduce head state to rank 0 (already on CPU — DDP keeps it in sync)
        else:
            full_sd = model.state_dict()
            head_sd = action_head.state_dict()

        if rank == 0:
            ckpt_dir = os.path.join(args.finetune_output, f"checkpoint_epoch{ep+1}")
            os.makedirs(ckpt_dir, exist_ok=True)

            lora_sd = {k: v for k, v in full_sd.items() if "lora_" in k}
            torch.save(lora_sd, os.path.join(ckpt_dir, "lora_weights.pt"))
            torch.save(head_sd, os.path.join(ckpt_dir, "action_head.pt"))
            if film_gen is not None:
                raw_film = film_gen.module if hasattr(film_gen, "module") else film_gen
                torch.save(raw_film.state_dict(), os.path.join(ckpt_dir, "film_weights.pt"))
            torch.save(
                {"optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "epoch":     ep + 1,
                 "hidden_size": hidden_size},
                os.path.join(ckpt_dir, "training_state.pt"),
            )
            log(f"[finetune] Checkpoint → {ckpt_dir}  "
                f"({len(lora_sd)} LoRA tensors + action head"
                f"{' + FiLM' if film_gen is not None else ''})")

            # Remove previous checkpoint to save disk space
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
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
            full_sd = model.state_dict()
        raw_head = action_head.module if hasattr(action_head, "module") else action_head
        head_sd  = raw_head.state_dict()
    else:
        full_sd = model.state_dict()
        head_sd = action_head.state_dict()

    if rank == 0:
        lora_sd = {k: v for k, v in full_sd.items() if "lora_" in k}
        torch.save(lora_sd, os.path.join(args.finetune_output, "lora_weights.pt"))
        torch.save(head_sd, os.path.join(args.finetune_output, "action_head.pt"))
        if film_gen is not None:
            raw_film = film_gen.module if hasattr(film_gen, "module") else film_gen
            torch.save(raw_film.state_dict(), os.path.join(args.finetune_output, "film_weights.pt"))
        # Save hidden_size so the inference server can auto-detect it
        torch.save({"hidden_size": hidden_size, "action_dim": args.action_dim,
                    "chunk_size": args.chunk_size, "lora_rank": args.lora_rank,
                    "lora_rank_vision": args.lora_rank_vision,
                    "use_film": args.use_film,
                    "film_hidden": args.film_hidden},
                   os.path.join(args.finetune_output, "meta.pt"))
        log(f"\n[finetune] Done.")
        log(f"[finetune]   LoRA   → {args.finetune_output}/lora_weights.pt")
        log(f"[finetune]   Head   → {args.finetune_output}/action_head.pt")
        log(f"[finetune]   Meta   → {args.finetune_output}/meta.pt")

    if world_size > 1:
        dist.destroy_process_group()


# ========================================================================== #
#  Inference server                                                            #
# ========================================================================== #

class OpenVLAOFTServer:
    """
    Loads OpenVLA-7B + LoRA + ActionChunkHead and runs OFT inference.

    TCP protocol identical to Pi0FastServer: length-prefixed JSON.
    """

    def __init__(
        self,
        model_dir:        str,
        lora_path:        str,
        head_path:        str,
        film_path:        str  = None,
        film_hidden:      int  = 256,
        chunk_size:       int  = 16,
        action_dim:       int  = 7,
        lora_rank:        int  = 32,
        lora_rank_vision: int  = 8,
        hidden_size:      int  = 4096,
        device:           str  = "cuda:0",
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")

        t0 = time.time()
        print(f"[OpenVLAOFTServer] Loading processor from '{model_dir}' …", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True, use_fast=True,
        )

        print(f"[OpenVLAOFTServer] Loading model from '{model_dir}' …", flush=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"[OpenVLAOFTServer] from_pretrained done in {time.time()-t0:.1f}s", flush=True)

        # ── Insert FiLM FIRST (changes block structure; must precede LoRA) ── #
        self.lang_holder  = [None]
        self.film_gen     = None
        self.embed_weight = None
        if film_path is not None and os.path.isfile(film_path):
            film_gen, self.lang_holder = insert_film(
                model, lang_dim=hidden_size,
                film_hidden=film_hidden,
                n_last_blocks=4,
            )
            if film_gen is not None:
                film_gen.load_state_dict(torch.load(film_path, map_location="cpu"))
                self.film_gen = film_gen.float().eval().to("cuda:0")
                self.film_gen.requires_grad_(False)
                self.embed_weight = model.language_model.model.embed_tokens.weight.detach().clone().to("cuda:0")
                print(f"[OpenVLAOFTServer] FiLM loaded from {film_path}", flush=True)

        # ── Apply LoRA structure (after FiLM so key paths match) ─────────── #
        t1 = time.time()
        n_llm = 0
        if lora_rank > 0:
            lora_alpha = lora_rank * 2
            llm_module = getattr(model, "language_model", model)
            n_llm = insert_lora(llm_module, target_suffixes=("q_proj", "k_proj", "v_proj", "o_proj"),
                                rank=lora_rank, alpha=lora_alpha)
        n_vis = 0
        if lora_rank_vision > 0:
            vb = getattr(model, "vision_backbone", None)
            if vb is not None:
                n_vis = insert_lora(vb, target_suffixes=("qkv", "proj"),
                                    rank=lora_rank_vision, alpha=lora_rank_vision * 2)
        print(f"[OpenVLAOFTServer] LoRA inserted: {n_llm} LLM + {n_vis} vision layers in "
              f"{time.time()-t1:.1f}s", flush=True)

        # ── Load LoRA weights ─────────────────────────────────────────── #
        t2 = time.time()
        lora_sd     = torch.load(lora_path, map_location="cpu")
        lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
        loaded, missing = 0, []
        for k, v in lora_sd.items():
            if k in lora_params:
                lora_params[k].data.copy_(v.half())
                loaded += 1
            else:
                missing.append(k)
        print(f"[OpenVLAOFTServer] LoRA loaded: {loaded}/{len(lora_sd)} tensors "
              f"in {time.time()-t2:.1f}s", flush=True)
        if missing:
            print(f"[OpenVLAOFTServer] WARNING: missing LoRA keys: {missing[:5]}")

        model.eval()
        model.requires_grad_(False)
        self.model = model
        # device_map="auto" places layers across GPUs — use first device for head/inputs
        self.device = next(model.parameters()).device

        # ── Load action head ──────────────────────────────────────────── #
        self.action_head = ActionChunkHead(hidden_size, action_dim, chunk_size)
        self.action_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        self.action_head = self.action_head.float().eval().to("cuda:0")
        self.action_head.requires_grad_(False)

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[OpenVLAOFTServer] {n_gpu} GPU(s).  Model ready.\n", flush=True)

    @torch.inference_mode()
    def predict(
        self,
        obs_rgb:   np.ndarray,
        instruction: str,
    ) -> list:
        """
        obs_rgb     : (H, W, 3) uint8
        instruction : task string
        Returns     : list of chunk_size lists of action_dim floats
        """
        pil_img = Image.fromarray(obs_rgb)
        prompt  = build_prompt(instruction)

        inputs = self.processor(
            text=prompt, images=pil_img,
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # Apply FiLM language conditioning if available
        if self.film_gen is not None and self.embed_weight is not None:
            text_enc = self.processor.tokenizer(
                instruction, return_tensors="pt", padding=True, truncation=True,
            ).to("cuda:0")
            text_emb = F.embedding(text_enc.input_ids, self.embed_weight.half())
            mask = text_enc.attention_mask.unsqueeze(-1).float()
            lang_emb = (text_emb.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            self.lang_holder[0] = self.film_gen(lang_emb)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        self.lang_holder[0] = None
        prefix_feat  = outputs.hidden_states[-1][:, -1, :].float().to("cuda:0")   # (1, hidden_size)
        pred_actions = self.action_head(prefix_feat)                               # (1, chunk_size, action_dim)

        return pred_actions[0].cpu().float().tolist()


# ========================================================================== #
#  TCP socket server  (same length-prefixed JSON protocol as pi0fast_server)  #
# ========================================================================== #

def run_server(args):
    """Resolve weights, start TCP server, serve inference requests."""
    import socket

    def resolve(path, filename):
        if os.path.isfile(path):
            return path
        candidate = os.path.join(path, filename)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"{filename} not found at '{path}'. "
            f"Pass --{filename.replace('.pt','').replace('_','')} as a .pt file "
            f"or a directory containing it."
        )

    lora_path = resolve(args.lora_path, "lora_weights.pt")
    head_path = resolve(args.head_path, "action_head.pt")

    # Auto-detect hidden_size, chunk_size, lora_rank from meta.pt if present
    meta_dir  = os.path.dirname(lora_path)
    meta_path = os.path.join(meta_dir, "meta.pt")
    # Also check parent directory (meta.pt saved at finetune_output root, not per-checkpoint)
    if not os.path.isfile(meta_path):
        meta_path = os.path.join(os.path.dirname(meta_dir), "meta.pt")
    hidden_size = args.hidden_size
    use_film    = args.use_film
    film_hidden = args.film_hidden
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        hidden_size            = meta.get("hidden_size",      hidden_size)
        args.chunk_size        = meta.get("chunk_size",       args.chunk_size)
        args.action_dim        = meta.get("action_dim",       args.action_dim)
        args.lora_rank         = meta.get("lora_rank",        args.lora_rank)
        args.lora_rank_vision  = meta.get("lora_rank_vision", args.lora_rank_vision)
        use_film               = meta.get("use_film",         False)
        film_hidden            = meta.get("film_hidden",      256)
        print(f"[run_server] meta.pt → hidden_size={hidden_size}  "
              f"chunk={args.chunk_size}  action_dim={args.action_dim}  "
              f"lora_rank={args.lora_rank}  lora_rank_vision={args.lora_rank_vision}  "
              f"use_film={use_film}", flush=True)

    if use_film:
        film_path = os.path.join(meta_dir, "film_weights.pt")
        if not os.path.isfile(film_path):
            film_path = os.path.join(os.path.dirname(meta_dir), "film_weights.pt")
    else:
        film_path = None

    server = OpenVLAOFTServer(
        model_dir        = args.model_dir,
        lora_path        = lora_path,
        head_path        = head_path,
        film_path        = film_path,
        film_hidden      = film_hidden,
        chunk_size       = args.chunk_size,
        action_dim       = args.action_dim,
        lora_rank        = args.lora_rank,
        lora_rank_vision = args.lora_rank_vision,
        hidden_size      = hidden_size,
        device           = args.device,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)

    print(f"\n[OpenVLAOFTServer] TCP server listening on {args.host}:{args.port}")
    print(f"[OpenVLAOFTServer] chunk_size={args.chunk_size}  — waiting for client …\n")

    n_requests = 0
    total_ms   = 0.0

    def handle_connection(conn, addr):
        nonlocal n_requests, total_ms
        print(f"[OpenVLAOFTServer] Client connected from {addr}")
        with conn:
            while True:
                raw_len = b""
                while len(raw_len) < 4:
                    chunk = conn.recv(4 - len(raw_len))
                    if not chunk:
                        print("[OpenVLAOFTServer] Client disconnected.")
                        return
                    raw_len += chunk

                req_len = int.from_bytes(raw_len, "big")
                chunks, received = [], 0
                while received < req_len:
                    chunk = conn.recv(min(65536, req_len - received))
                    if not chunk:
                        print("[OpenVLAOFTServer] Client disconnected mid-message.")
                        return
                    chunks.append(chunk)
                    received += len(chunk)

                data = json.loads(b"".join(chunks))
                seq  = data.get("seq", 0)

                try:
                    obs_rgb = np.array(
                        Image.open(io.BytesIO(base64.b64decode(data["jpeg_b64"]))).convert("RGB"),
                        dtype=np.uint8,
                    )
                except Exception as e:
                    resp    = {"seq": seq, "action": [[0.0] * args.action_dim],
                               "status": "error", "message": f"Image decode error: {e}"}
                    payload = json.dumps(resp).encode()
                    conn.sendall(len(payload).to_bytes(4, "big") + payload)
                    continue

                instruction = data.get("instruction", args.instruction)

                if n_requests == 0:
                    print(f"[DEBUG] Client image: shape={obs_rgb.shape} dtype={obs_rgb.dtype} "
                          f"min={obs_rgb.min()} max={obs_rgb.max()} mean={obs_rgb.mean():.1f}",
                          flush=True)
                    print(f"[DEBUG] Client instruction: \"{instruction}\"", flush=True)
                    print(f"[DEBUG] Train  instruction: \"pick up the red cube and place it on the tray\"",
                          flush=True)

                t0           = time.perf_counter()
                action_chunk = server.predict(obs_rgb, instruction)
                ms           = (time.perf_counter() - t0) * 1000

                if n_requests < 3:
                    step0 = action_chunk[0]
                    print(f"[DEBUG] action_chunk[0] = [{', '.join(f'{v:+.4f}' for v in step0)}]",
                          flush=True)

                n_requests += 1
                total_ms   += ms

                # Always send full chunk; exec_horizon passed to client to decide how many to execute
                exec_horizon = args.exec_horizon if args.exec_horizon else len(action_chunk)
                response = {"seq": seq, "action": action_chunk, "exec_horizon": exec_horizon, "status": "ok"}
                payload  = json.dumps(response).encode()
                conn.sendall(len(payload).to_bytes(4, "big") + payload)

                if n_requests % 20 == 0:
                    print(f"[OpenVLAOFTServer] {n_requests} requests  "
                          f"avg_latency={total_ms/n_requests:.1f} ms")

    try:
        while True:
            conn, addr = sock.accept()
            handle_connection(conn, addr)
    except KeyboardInterrupt:
        print(f"\n[OpenVLAOFTServer] Shutting down after {n_requests} requests.")
    finally:
        sock.close()


# ========================================================================== #
#  CLI                                                                         #
# ========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenVLA-OFT: fine-tuning and inference on Franka demonstrations"
    )

    # Mode
    p.add_argument("--serve", action="store_true",
                   help="Run inference server instead of fine-tuning")

    # Model
    p.add_argument("--model_dir", type=str, default="openvla/openvla-7b",
                   help="HuggingFace hub ID or local path to OpenVLA-7B weights")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint_epochN directory to resume fine-tuning")
    p.add_argument("--reset_optimizer", action="store_true",
                   help="Resume weights only; restart optimizer/scheduler from scratch")
    p.add_argument("--finetune_output", type=str,
                   default="outputs/openvla_oft_finetuned")

    # Inference server weights
    p.add_argument("--lora_path", type=str,
                   default="outputs/openvla_oft_finetuned/lora_weights.pt",
                   help="Path to lora_weights.pt (or directory containing it)")
    p.add_argument("--head_path", type=str,
                   default="outputs/openvla_oft_finetuned/action_head.pt",
                   help="Path to action_head.pt (or directory containing it)")
    p.add_argument("--hidden_size", type=int, default=4096,
                   help="LLM hidden size (auto-detected from meta.pt when serving)")

    # Data — multi-task (--tasks) or single-task (--demo_data)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Task name(s) or 'all'. Reads data/<task>/demos.hdf5 + task.json.")
    p.add_argument("--data_dir", type=str, default="data",
                   help="Root directory containing per-task folders (used with --tasks)")
    p.add_argument("--demo_data", type=str,
                   default="data/pick_red_cube_to_tray/demos.hdf5",
                   help="Single HDF5 path for single-task fine-tuning")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--episodes_per_task", type=int, default=25,
                   help="Episodes sampled per task per epoch (lazy resampling)")

    # Training
    p.add_argument("--chunk_size",  type=int,   default=16,
                   help="Action chunk length (OFT parallel prediction horizon)")
    p.add_argument("--exec_horizon", type=int, default=None,
                   help="Steps to execute per chunk (default: full chunk_size). Set to 8 to replan more frequently.")
    p.add_argument("--action_dim",  type=int,   default=7,
                   help="Action space dimension (7 for Franka)")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--instruction", type=str,
                   default="pick up the red cube and place it on the tray",
                   help="Instruction for single-task fine-tuning / server fallback")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=32,
                   help="LoRA rank for Q/K/V/O projections (default 32)")
    p.add_argument("--lora_rank_vision", type=int, default=8,
                   help="LoRA rank for vision backbone attn layers (default 8)")
    p.add_argument("--vision_lora_last_blocks", type=int, default=4,
                   help="Number of last blocks in each vision encoder to apply LoRA to. "
                        "Fewer blocks = shallower fp16 gradient path = more stable (default 4)")

    # FiLM
    p.add_argument("--use_film", action="store_true",
                   help="Insert FiLM conditioning into vision encoder blocks (OFT+ style)")
    p.add_argument("--film_hidden", type=int, default=256,
                   help="Hidden dim of FiLM generator MLP (default 256)")

    # Transition-focused sampling
    p.add_argument("--non_transition_prob", type=float, default=0.0,
                   help="Fraction of non-transition chunks to KEEP (0.0=keep all [off], 0.1=keep 10%%, 1.0=keep all). "
                        "Use e.g. 0.1 to heavily enrich transition chunks in the dataset.")
    p.add_argument("--transition_lookahead", type=int, default=16,
                   help="Steps ahead to look for a gripper transition when labelling/upweighting chunks (default=16=chunk_size). "
                        "Set to 32 to upweight chunks up to 32 steps before a gripper transition.")

    # Server
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--host",   type=str, default="127.0.0.1")
    p.add_argument("--port",   type=int, default=5556,
                   help="TCP port (default 5556 — avoids clash with pi0fast_server on 5555)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.serve:
        run_server(args)
    else:
        finetune_openvla_oft(args)
