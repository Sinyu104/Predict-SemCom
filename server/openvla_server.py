"""
openvla_server.py  —  OpenVLA discrete-token fine-tuning and TCP inference.

Trains OpenVLA-7B with LoRA (rank=32, alpha=64, q/k/v/o) using cross-entropy
on 7 discrete action tokens (256-bin quantized).  Cross-entropy is mode-seeking,
unlike OFT's MSE head which collapses to mean trajectories and fails at gripper
transitions.  This server achieved 83% success rate in earlier testing.

Architecture
------------
  Base model  : openvla/openvla-7b  (SigLIP-400M + DINOv2-L + Llama-2-7B)
  LoRA        : custom rank=32, alpha=64 on q/k/v/o in the LLM (no peft dep)
  Action head : VLM autoregressive 7-token decoding (256 bins, mode-seeking)
  Loss        : teacher-forcing CE on 7 action tokens appended to the prompt (official OpenVLA)

vs openvla_oft_server.py
-------------------------
  - No custom MLP head; predict_action() decodes tokens autoregressively
  - Custom LoRALinear (same as OFT) instead of peft, avoids peft/transformers
    version clash in the worldmodel conda env
  - Single-process training with device_map="auto" across all available GPUs, float16 + autocast
  - Dataset returns single-step (image, instruction, action); exec_horizon=1
  - Checkpoint: lora_weights.pt + processor files + dataset_statistics.json

Inference also supports loading the earlier PEFT safetensors checkpoint
(outputs/openvla_finetuned_v3) via automatic key-stripping.

franka_isaac statistics: mean=0, std=1, q01=-1, q99=+1 (identity normalisation)

Usage
-----
Fine-tune (single process, all GPUs via device_map="auto"):
    python server/openvla_server.py \\
        --tasks all --data_dir data \\
        --finetune_output outputs/openvla_discrete \\
        --epochs 20 --batch_size 4

Resume from checkpoint:
    python server/openvla_server.py \\
        --resume_from outputs/openvla_discrete/checkpoint_epoch5 \\
        --tasks all --finetune_output outputs/openvla_discrete --epochs 20

Inference from new checkpoint (lora_weights.pt format):
    python server/openvla_server.py --serve \\
        --lora_path outputs/openvla_discrete/checkpoint_epoch6

Inference from 83% PEFT checkpoint (adapter_model.safetensors format):
    python server/openvla_server.py --serve \\
        --lora_path outputs/openvla_finetuned_v3/checkpoint_epoch6

HDF5 layout:
    data/<task>/demos.hdf5
      /episode_N/observations_cam1  (T, 224, 224, 3)  uint8
      /episode_N/actions            (T, 7)             float32  in [-1, 1]
    data/<task>/task.json  →  {"instruction": "..."}
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

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

UNNORM_KEY = "franka_isaac"

# Per-dimension action statistics computed over all 114,407 frames (5 tasks, ~99 eps each).
# Used for BOUNDS_Q99 normalization (official OpenVLA): each dim is scaled
# [q01, q99] → [-1, 1] before binning, so all 256 bins are used per dim.
# Rotation dims (roll/pitch/yaw) are always 0 → q01=q99=0 (handled by +1e-8).
DATASET_STATISTICS = {
    UNNORM_KEY: {
        "action": {
            "mean": [0.03955, -0.08149, -0.08731, 0.0, 0.0, 0.0, -0.23423],
            "std":  [0.06183,  0.13193,  0.07252, 0.0, 0.0, 0.0,  0.97207],
            "max":  [0.29807,  0.29345,  0.11703, 0.0, 0.0, 0.0,  1.0],
            "min":  [-0.16594, -0.69394, -0.37595, 0.0, 0.0, 0.0, -1.0],
            "q01":  [-0.10267, -0.57834, -0.28638, 0.0, 0.0, 0.0, -1.0],
            "q99":  [0.24051,   0.15324,  0.08683, 0.0, 0.0, 0.0,  1.0],
        }
    }
}

# Arrays for fast normalization in tokenize_action (BOUNDS_Q99).
_ACTION_Q01 = np.array(DATASET_STATISTICS[UNNORM_KEY]["action"]["q01"], dtype=np.float64)
_ACTION_Q99 = np.array(DATASET_STATISTICS[UNNORM_KEY]["action"]["q99"], dtype=np.float64)

ACTION_TOKEN_COUNT = 7   # OpenVLA encodes each action as 7 discrete tokens

# OpenVLA action-token encoding constants — matches official ActionTokenizer:
#   n_bins = 256, bins = linspace(-1,1,257) (257 edges → 256 intervals)
#   np.digitize returns [1, 256]; token_id = vocab_size - digitized  → [31744, 31999]
_OPENVLA_VOCAB_SIZE  = 32000
_N_ACTION_BINS       = 256
_OPENVLA_BIN_EDGES   = np.linspace(-1.0, 1.0, _N_ACTION_BINS + 1)       # 257 edges

# Action token range: [31744, 31999]
ACTION_TOKEN_START = _OPENVLA_VOCAB_SIZE - _N_ACTION_BINS    # 31744
ACTION_TOKEN_END   = _OPENVLA_VOCAB_SIZE                     # 32000
N_ACTION_BINS      = _N_ACTION_BINS                          # 256


# ---------------------------------------------------------------------------
#  Custom LoRA linear (same as openvla_oft_server.py, no peft dependency)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: int):
        super().__init__()
        dev   = linear.weight.device
        dtype = linear.weight.dtype
        in_f, out_f = linear.in_features, linear.out_features
        self.weight = linear.weight
        self.bias   = getattr(linear, "bias", None)
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=dev, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dtype))
        self.scale  = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lA  = self.lora_A if self.lora_A.dtype == x.dtype else self.lora_A.to(x.dtype)
        lB  = self.lora_B if self.lora_B.dtype == x.dtype else self.lora_B.to(x.dtype)
        return out + F.linear(F.linear(x, lA), lB) * self.scale


def insert_lora(
    model: nn.Module,
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 32,
    alpha: int = 64,
) -> int:
    """Replace target nn.Linear layers with LoRALinear in-place.  Returns count."""
    replaced = 0
    for full_name, module in list(model.named_modules()):
        for suffix in target_suffixes:
            if full_name.endswith(f".{suffix}") and isinstance(module, nn.Linear):
                parent_name, attr = full_name.rsplit(".", 1)
                parent = model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                setattr(parent, attr, LoRALinear(module, rank, alpha))
                replaced += 1
                break
    return replaced


# ---------------------------------------------------------------------------
#  FiLM (Feature-wise Linear Modulation) for vision-language conditioning.
#  Ported from openvla_oft_server.py — strengthens color grounding by letting
#  the instruction directly modulate the last N vision-encoder blocks.
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """Mean-pooled instruction embedding → per-block (gamma, beta). Zero-init → identity."""
    def __init__(self, lang_dim: int, film_hidden: int, feat_dims: list):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(lang_dim, film_hidden), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(film_hidden, d * 2) for d in feat_dims])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, lang_emb: torch.Tensor) -> list:
        h = self.shared(lang_emb.float())
        return [head(h).chunk(2, dim=-1) for head in self.heads]


class FiLMBlock(nn.Module):
    """Wraps a vision block; applies FiLM after it using gamma/beta from lang_holder[0]."""
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
            gamma = gamma.unsqueeze(1).to(device=x.device, dtype=x.dtype)   # (B,1,feat)
            beta  = beta.unsqueeze(1).to(device=x.device, dtype=x.dtype)
            x = (1.0 + gamma) * x + beta                                    # identity at zero-init
        return x


def insert_film(model: nn.Module, lang_dim: int,
                film_hidden: int = 256, n_last_blocks: int = 4):
    """Wrap last n_last_blocks of each vision encoder with FiLMBlock.
    Returns (film_gen, lang_holder); film_gen is None if no vision backbone found."""
    lang_holder = [None]
    vb = getattr(model, "vision_backbone", None)
    if vb is None:
        return None, lang_holder

    target_blocks, feat_dims = [], []
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


# ---------------------------------------------------------------------------
#  Dataset — single-step (image, instruction, action)
# ---------------------------------------------------------------------------

class MultiTaskStepDataset(TorchDataset):
    """
    Lazy-loading multi-task dataset.
    Each sample is one timestep: (image, instruction, action).
    Call resample(epoch) at the start of each epoch.
    """

    def __init__(
        self,
        task_dirs:            list,
        n_episodes_per_task:  int = 25,
        frame_stride:         int = 1,
        non_transition_prob:  float = 0.0,
        transition_lookahead: int = 16,
        dagger_episodes_per_task: int = 0,
    ):
        self.n_per_task           = n_episodes_per_task
        self.frame_stride         = frame_stride
        self.non_transition_prob  = non_transition_prob
        self.transition_lookahead = transition_lookahead
        self._task_meta   = []
        self._handles     = {}
        self.samples      = []

        for task_dir in task_dirs:
            json_ = os.path.join(task_dir, "task.json")
            if not os.path.isfile(os.path.join(task_dir, "demos.hdf5")):
                print(f"[Dataset] Skipping {task_dir}: no demos.hdf5")
                continue
            if not os.path.isfile(json_):
                raise FileNotFoundError(f"task.json missing in {task_dir}")
            with open(json_) as f:
                instr = json.load(f).get("instruction", "")

            # DAgger aggregation: register dagger.hdf5 as a second pool for this
            # task, with its own per-epoch episode cap, so each epoch mixes
            # expert demos with on-policy corrections (D <- D u D_new).
            # Training on the DAgger file ALONE dropped success from 45% to 10%
            # (2026-08, epochs 24/25) — DAgger data is failure-states only, so
            # without the demos the policy has no successful grasp to imitate.
            sources = [("demos.hdf5", n_episodes_per_task)]
            if dagger_episodes_per_task > 0:
                sources.append(("dagger.hdf5", dagger_episodes_per_task))

            for fname, n_cap in sources:
                hdf5 = os.path.join(task_dir, fname)
                if not os.path.isfile(hdf5):
                    print(f"[Dataset] {task_dir}: no {fname}, skipping that pool")
                    continue
                with h5py.File(hdf5, "r") as f:
                    ep_keys = sorted(
                        [k for k in f.keys()
                         if "observations_cam1" in f[k] and "actions" in f[k]],
                        key=lambda k: int(k.split("_")[1]),
                    )
                    ep_len = {k: len(f[k]["actions"]) for k in ep_keys}
                self._task_meta.append((hdf5, instr, ep_keys, ep_len, n_cap))
                print(f"[Dataset] {hdf5}: {len(ep_keys)} eps  "
                      f"cap={n_cap}/epoch  instr={instr!r:.60s}")

        print(f"[Dataset] {len(self._task_meta)} tasks. Call resample(epoch).")

    def resample(self, epoch: int):
        rng = np.random.default_rng(epoch)
        self.samples = []
        for hdf5, instr, ep_keys, ep_len, n_cap in self._task_meta:
            n = min(n_cap, len(ep_keys))
            chosen = rng.choice(ep_keys, size=n, replace=False)
            for ep in chosen:
                for t in range(0, ep_len[ep], self.frame_stride):
                    self.samples.append((hdf5, ep, t, instr))

        # Gripper-transition oversampling: keep ALL frames within transition_lookahead
        # of a gripper flip; keep the rest with prob=non_transition_prob. This rebalances
        # the ~0.9% transition frames up to a meaningful fraction of the dataset.
        if 0.0 < self.non_transition_prob < 1.0:
            keep = []
            n_transition = 0
            rand_vals = rng.random(len(self.samples))
            for i, (hdf5, ep, t, instr) in enumerate(self.samples):
                f = self._h(hdf5)
                grip = np.array(f[ep]["actions"][t : t + self.transition_lookahead, 6])
                is_transition = grip.size > 1 and (np.abs(np.diff(grip)) > 0.5).any()
                if is_transition:
                    keep.append(self.samples[i]); n_transition += 1
                elif rand_vals[i] < self.non_transition_prob:
                    keep.append(self.samples[i])
            self.samples = keep
            print(f"[Dataset] non_transition_prob={self.non_transition_prob:.2f}: "
                  f"{n_transition} transition kept of {len(self.samples)} total "
                  f"({100*n_transition/max(len(self.samples),1):.1f}% transition)")

        rng.shuffle(self.samples)
        n_dagger = sum(1 for s in self.samples if s[0].endswith("dagger.hdf5"))
        extra = (f", {n_dagger} from dagger.hdf5 "
                 f"({100*n_dagger/max(len(self.samples),1):.1f}%)") if n_dagger else ""
        print(f"[Dataset] epoch={epoch}: {len(self.samples)} samples "
              f"from {len(self._task_meta)} pools{extra}")

    def _h(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5, ep, t, instr = self.samples[idx]
        f = self._h(hdf5)
        img    = torch.from_numpy(
            np.array(f[ep]["observations_cam1"][t])
        ).permute(2, 0, 1).float() / 255.0           # (3, H, W)
        action = torch.from_numpy(
            np.array(f[ep]["actions"][t], dtype=np.float32)
        )                                              # (7,)
        return img, instr, action


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def build_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def register_statistics(processor=None, model=None):
    """
    Inject franka_isaac norm_stats into the model (for predict_action) and/or
    into the processor's action_tokenizer (if it has one).
    """
    if model is not None and hasattr(model, "norm_stats"):
        model.norm_stats.update(DATASET_STATISTICS)
    at = getattr(processor, "action_tokenizer", None) if processor is not None else None
    if at is not None:
        if hasattr(at, "dataset_statistics"):
            at.dataset_statistics.update(DATASET_STATISTICS)
        else:
            at.dataset_statistics = dict(DATASET_STATISTICS)


def tokenize_action(action_np: np.ndarray, vocab_size: int = _OPENVLA_VOCAB_SIZE) -> torch.Tensor:
    """
    Encode a (7,) float32 action → (7,) int64 OpenVLA token IDs.

    Matches the official OpenVLA pipeline exactly:
      1. BOUNDS_Q99 normalize each dim: clip(2*(x-q01)/(q99-q01+1e-8) - 1, -1, 1)
         → each dim's [q01, q99] range fills [-1, 1], using all 256 bins.
      2. digitize over _OPENVLA_BIN_EDGES → [1, 256]; token_id = vocab_size - digitized.

    Inference un-normalizes via predict_action + unnorm_key (DATASET_STATISTICS).
    """
    norm = 2.0 * (action_np - _ACTION_Q01) / (_ACTION_Q99 - _ACTION_Q01 + 1e-8) - 1.0
    norm = np.clip(norm, -1.0, 1.0)
    digitized  = np.digitize(norm, _OPENVLA_BIN_EDGES)                    # [1, 257]
    digitized  = np.clip(digitized, 1, _N_ACTION_BINS)                    # clamp to [1, 256]
    token_ids  = vocab_size - digitized                                    # [31744, 31999]
    return torch.tensor(token_ids, dtype=torch.long)


def save_dataset_statistics(directory: str):
    with open(os.path.join(directory, "dataset_statistics.json"), "w") as f:
        json.dump(DATASET_STATISTICS, f, indent=2)


# ---------------------------------------------------------------------------
#  LoRA weight loading: supports both custom .pt and PEFT safetensors
# ---------------------------------------------------------------------------

def load_lora_weights(model: nn.Module, lora_dir: str):
    """
    Load LoRA weights into a model that already has LoRALinear layers inserted.

    Supports two formats:
    1. Custom: lora_weights.pt  (keys: model-relative, contain "lora_")
    2. PEFT:   adapter_model.safetensors  (keys prefixed with "base_model.model.")
    """
    custom_pt = os.path.join(lora_dir, "lora_weights.pt")
    peft_st   = os.path.join(lora_dir, "adapter_model.safetensors")

    if os.path.isfile(custom_pt):
        sd = torch.load(custom_pt, map_location="cpu")
        lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
        loaded = sum(
            1 for k, v in sd.items()
            if k in lora_params and not lora_params[k].data.copy_(v.to(lora_params[k].dtype)) is None
        )
        print(f"[load_lora] Custom pt: {loaded}/{len(sd)} tensors from {custom_pt}")
        return loaded

    if os.path.isfile(peft_st):
        from safetensors.torch import load_file
        sd = load_file(peft_st)
        # PEFT keys: "base_model.model.<model_attr_path>.lora_A.weight"
        # Strip "base_model.model." prefix to get model-relative names.
        # Then map to our LoRALinear: "...q_proj.lora_A.weight" → lora_A param
        lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
        prefix = "base_model.model."
        mapped, loaded = {}, 0
        for k, v in sd.items():
            k_stripped = k[len(prefix):] if k.startswith(prefix) else k
            # PEFT: ".lora_A.weight" → LoRALinear: ".lora_A" (no .weight suffix)
            # The LoRALinear parameter name in named_parameters() is e.g.
            # "language_model.model.layers.0.self_attn.q_proj.lora_A"
            k_mapped = k_stripped.replace(".lora_A.weight", ".lora_A") \
                                  .replace(".lora_B.weight", ".lora_B")
            mapped[k_mapped] = v
        for k, v in mapped.items():
            if k in lora_params:
                lora_params[k].data.copy_(v.to(lora_params[k].dtype))
                loaded += 1
        print(f"[load_lora] PEFT safetensors: {loaded}/{len(sd)} tensors from {peft_st}")
        return loaded

    raise FileNotFoundError(
        f"Neither lora_weights.pt nor adapter_model.safetensors found in {lora_dir!r}"
    )


# ---------------------------------------------------------------------------
#  Fine-tuning
# ---------------------------------------------------------------------------

def finetune(args):
    """
    Fine-tune OpenVLA-7B with custom LoRA, cross-entropy on action tokens.
    Single process; device_map="auto" spreads the model across all GPUs.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # ── Resume epoch ────────────────────────────────────────────────── #
    start_epoch = 0
    if args.resume_from:
        m = re.match(r".*checkpoint_epoch(\d+)$", args.resume_from.rstrip("/"))
        if not m:
            raise ValueError(f"--resume_from must end with checkpoint_epochN, "
                             f"got {args.resume_from!r}")
        start_epoch = int(m.group(1))
        print(f"\n[finetune] Resuming from {args.resume_from} (epoch {start_epoch})")
    else:
        print(f"\n[finetune] Base model: {args.model_dir}")

    total_epochs = start_epoch + args.epochs
    os.makedirs(args.finetune_output, exist_ok=True)

    # ── Processor ───────────────────────────────────────────────────── #
    proc_src = args.resume_from if args.resume_from else args.model_dir
    print(f"[finetune] Loading processor from {proc_src} …")
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)
    # ── Base model ───────────────────────────────────────────────────── #
    print(f"[finetune] Loading model (device_map=auto) …")
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    register_statistics(processor=processor, model=model)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Insert LoRA ──────────────────────────────────────────────────── #
    # "all-linear" matches official OpenVLA: attention + MLP projections
    llm = getattr(model, "language_model", model)
    n_lora = insert_lora(
        llm,
        target_suffixes=("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print(f"[finetune] Inserted LoRA: {n_lora} layers  "
          f"(rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"{time.time()-t0:.1f}s)")

    if args.resume_from:
        loaded = load_lora_weights(model, args.resume_from)
        print(f"[finetune] Resumed {loaded} LoRA tensors from {args.resume_from}")

    # Cast LoRA params to FP32 so GradScaler can unscale them correctly.
    # Frozen weights stay in FP16; LoRALinear.forward() casts A/B on the fly.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    first_device = next(model.parameters()).device

    # ── FiLM language conditioning ──────────────────────────────────── #
    film_gen, lang_holder = None, [None]
    embed_weight = None
    if args.use_film:
        embed_weight = model.language_model.model.embed_tokens.weight.detach().clone()
        lang_dim = embed_weight.shape[1]
        film_gen, lang_holder = insert_film(
            model, lang_dim=lang_dim,
            film_hidden=args.film_hidden, n_last_blocks=4,
        )
        if film_gen is not None:
            film_gen = film_gen.float().to(first_device)
            embed_weight = embed_weight.to(first_device)
            if args.resume_from:
                fpt = os.path.join(args.resume_from, "film_weights.pt")
                if os.path.isfile(fpt):
                    film_gen.load_state_dict(torch.load(fpt, map_location="cpu"))
                    print(f"[finetune] Resumed FiLM weights from {fpt}")
            n_film = sum(p.numel() for p in film_gen.parameters())
            print(f"[finetune] FiLM inserted: {len(film_gen.heads)} blocks, "
                  f"{n_film:,} params (film_hidden={args.film_hidden})")
        else:
            print("[finetune] WARNING: --use_film set but no vision backbone found; FiLM disabled")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] Trainable: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.3f}%)")

    # ── Dataset ─────────────────────────────────────────────────────── #
    task_names = ALL_TASKS if args.tasks == ["all"] else args.tasks
    task_dirs  = [os.path.join(args.data_dir, t) for t in task_names]
    dataset    = MultiTaskStepDataset(
        task_dirs=task_dirs,
        n_episodes_per_task=args.episodes_per_task,
        frame_stride=args.frame_stride,
        non_transition_prob=args.non_transition_prob,
        transition_lookahead=args.transition_lookahead,
        dagger_episodes_per_task=args.dagger_episodes_per_task,
    )

    # ── Optimizer ───────────────────────────────────────────────────── #
    # Constant lr matches official OpenVLA (no scheduler)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    film_params = list(film_gen.parameters()) if film_gen is not None else []
    optimizer   = torch.optim.AdamW(lora_params + film_params, lr=args.lr)
    scaler      = torch.cuda.amp.GradScaler()   # required for float16 stability

    # Optional cosine LR decay over the (resumed) epoch span — lets a bouncing
    # constant-lr loss settle into the minimum. T_max = remaining epochs.
    scheduler = None
    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - start_epoch),
            eta_min=args.lr * args.lr_min_ratio,
        )
        print(f"[finetune] Cosine LR decay ON: {args.lr:.2e} → "
              f"{args.lr*args.lr_min_ratio:.2e} over {total_epochs-start_epoch} epochs")

    if args.resume_from and not args.reset_optimizer:
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            try:
                optimizer.load_state_dict(state["optimizer"])
                print(f"[finetune] Optimizer resumed")
            except (ValueError, KeyError) as e:
                print(f"[finetune] Optimizer state mismatch ({e}), starting fresh")

    # ── Training loop ─────────────────────────────────────────────── #
    for ep in range(start_epoch, total_epochs):
        dataset.resample(ep)
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=0, drop_last=False,
        )

        model.train()
        total_loss  = 0.0
        n_steps     = 0
        nan_skipped = 0

        step_bar = tqdm(loader, desc=f"  Epoch {ep+1}/{total_epochs}",
                        unit="batch", leave=False)

        for imgs, instructions, actions in step_bar:
            B          = imgs.size(0)
            pil_imgs   = [tensor_to_pil(imgs[i]) for i in range(B)]
            prompts    = [build_prompt(instructions[i]) for i in range(B)]
            action_ids = [tokenize_action(actions[i].numpy()) for i in range(B)]
            # action_ids: list of B (7,) long tensors

            try:
                # Batch processor call (left-pads shorter prompts)
                inputs = processor(
                    text=prompts, images=pil_imgs,
                    return_tensors="pt", padding=True, truncation=True,
                )
            except Exception as e:
                print(f"\n[finetune] Processor error ({e}), skipping batch")
                continue

            # Stack action token IDs: (B, 7)
            act_batch = torch.stack(action_ids)   # (B, 7)

            # Teacher forcing — matches official OpenVLA finetune.py:
            # append action tokens to prompt, mask prompt positions with -100.
            full_ids = torch.cat([inputs.input_ids, act_batch], dim=1)       # (B, T+7)
            full_attn = torch.cat([
                inputs.attention_mask,
                torch.ones(B, ACTION_TOKEN_COUNT, dtype=torch.long),
            ], dim=1)                                                          # (B, T+7)
            labels = torch.full_like(full_ids, -100)
            labels[:, -ACTION_TOKEN_COUNT:] = act_batch

            pv = inputs.pixel_values.to(dtype=torch.float16, device=first_device)

            # FiLM: compute mean-pooled instruction embedding → (gamma, beta) per block.
            if film_gen is not None and embed_weight is not None:
                text_enc = processor.tokenizer(
                    list(instructions), return_tensors="pt",
                    padding=True, truncation=True,
                ).to(first_device)
                with torch.no_grad():
                    text_emb = F.embedding(text_enc.input_ids, embed_weight.half())
                mask = text_enc.attention_mask.unsqueeze(-1).float()
                lang_emb = (text_emb.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                lang_holder[0] = film_gen(lang_emb)

            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(
                        input_ids      = full_ids.to(first_device),
                        pixel_values   = pv,
                        attention_mask = full_attn.to(first_device),
                        labels         = labels.to(first_device),
                    )
            except Exception as e:
                import traceback
                lang_holder[0] = None
                print(f"\n[finetune] Forward error:\n{traceback.format_exc()}")
                continue
            lang_holder[0] = None   # clear after forward

            if args.gripper_loss_weight != 1.0:
                # Per-token weighted CE: up-weight the gripper token (dim 6, the
                # 7th/last action token) so the model learns its sharp close-timing.
                # HF shift: logits[-8:-1] predict the 7 action tokens (labels[-7:]).
                V = out.logits.size(-1)
                shift_logits = out.logits[:, -(ACTION_TOKEN_COUNT + 1):-1, :]      # (B,7,V)
                target = act_batch.to(first_device)                               # (B,7)
                ce = F.cross_entropy(
                    shift_logits.reshape(-1, V), target.reshape(-1),
                    reduction="none",
                ).view(-1, ACTION_TOKEN_COUNT)                                     # (B,7)
                w = torch.ones(ACTION_TOKEN_COUNT, device=ce.device)
                w[6] = args.gripper_loss_weight
                loss = (ce * w).sum() / (ce.size(0) * w.sum())   # weighted mean, scale-matched
            else:
                loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_skipped += 1
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params + film_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_steps    += 1
            step_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        print(f"[finetune] Epoch {ep+1}/{total_epochs}  "
              f"loss={avg_loss:.4f}  lr={cur_lr:.2e}  "
              f"nan_skipped={nan_skipped}", flush=True)
        if scheduler is not None:
            scheduler.step()

        # ── Checkpoint ──────────────────────────────────────────────── #
        ckpt_dir = os.path.join(args.finetune_output, f"checkpoint_epoch{ep+1}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save LoRA weights
        lora_sd = {n: p.detach().cpu() for n, p in model.named_parameters()
                   if "lora_" in n}
        torch.save(lora_sd, os.path.join(ckpt_dir, "lora_weights.pt"))

        # Save FiLM weights + meta
        if film_gen is not None:
            torch.save(film_gen.state_dict(), os.path.join(ckpt_dir, "film_weights.pt"))
        with open(os.path.join(ckpt_dir, "film_meta.json"), "w") as f:
            json.dump({"use_film": bool(film_gen is not None),
                       "film_hidden": args.film_hidden}, f)

        # Save processor (tokenizer + preprocessor_config)
        processor.save_pretrained(ckpt_dir)
        save_dataset_statistics(ckpt_dir)

        torch.save(
            {"optimizer": optimizer.state_dict(),
             "epoch":     ep + 1},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        print(f"[finetune] Checkpoint → {ckpt_dir}", flush=True)

        if ep > start_epoch:
            prev = os.path.join(args.finetune_output, f"checkpoint_epoch{ep}")
            if os.path.isdir(prev):
                shutil.rmtree(prev)

    print(f"\n[finetune] Done.  Checkpoint → {ckpt_dir}")


# ---------------------------------------------------------------------------
#  Inference server
# ---------------------------------------------------------------------------

class OpenVLAServer:
    """
    Loads OpenVLA-7B + custom LoRA and serves TCP inference requests.

    Supports loading from:
    - lora_weights.pt  (produced by this script)
    - adapter_model.safetensors  (PEFT format, e.g. outputs/openvla_finetuned_v3)
    """

    def __init__(self, lora_dir: str, model_dir: str):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Load processor from checkpoint dir (has tokenizer.json) or base model
        proc_src = lora_dir if os.path.isfile(
            os.path.join(lora_dir, "tokenizer.json")
        ) else model_dir
        print(f"[OpenVLAServer] Processor from {proc_src} …", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            proc_src, trust_remote_code=True
        )
        print(f"[OpenVLAServer] Base model from {model_dir} (device_map=auto) …",
              flush=True)
        t0 = time.time()
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        register_statistics(processor=self.processor, model=self.model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Insert LoRA structure — MUST match training (all-linear, alpha=16)
        llm = getattr(self.model, "language_model", self.model)
        n   = insert_lora(
            llm,
            target_suffixes=("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"),
            rank=32, alpha=16,
        )
        print(f"[OpenVLAServer] Inserted {n} LoRA layers", flush=True)

        # Load weights
        load_lora_weights(self.model, lora_dir)

        self.model.eval()
        self._device = next(self.model.parameters()).device

        # ── FiLM (if the checkpoint was trained with it) ──────────────── #
        self.film_gen, self.lang_holder = None, [None]
        self.embed_weight = None
        film_pt   = os.path.join(lora_dir, "film_weights.pt")
        film_meta = os.path.join(lora_dir, "film_meta.json")
        if os.path.isfile(film_pt):
            film_hidden = 256
            if os.path.isfile(film_meta):
                with open(film_meta) as f:
                    film_hidden = json.load(f).get("film_hidden", 256)
            self.embed_weight = self.model.language_model.model.embed_tokens.weight.detach().clone()
            lang_dim = self.embed_weight.shape[1]
            self.film_gen, self.lang_holder = insert_film(
                self.model, lang_dim=lang_dim, film_hidden=film_hidden, n_last_blocks=4,
            )
            self.film_gen = self.film_gen.float().to(self._device)
            self.embed_weight = self.embed_weight.to(self._device)
            self.film_gen.load_state_dict(torch.load(film_pt, map_location="cpu"))
            self.film_gen.eval()
            print(f"[OpenVLAServer] FiLM loaded: {len(self.film_gen.heads)} blocks", flush=True)

        print(f"[OpenVLAServer] Ready in {time.time()-t0:.1f}s  "
              f"({torch.cuda.device_count()} GPU(s))\n", flush=True)

    @torch.inference_mode()
    def predict(self, obs_rgb: np.ndarray, instruction: str) -> list:
        """
        obs_rgb     : (H, W, 3) uint8
        instruction : task string
        Returns     : list of 7 floats (single-step action in [-1, 1])
        """
        pil_img = Image.fromarray(obs_rgb)
        prompt  = build_prompt(instruction)
        inputs  = self.processor(
            text=prompt, images=pil_img, return_tensors="pt"
        ).to(self._device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # FiLM language conditioning
        if self.film_gen is not None and self.embed_weight is not None:
            text_enc = self.processor.tokenizer(
                instruction, return_tensors="pt", padding=True, truncation=True,
            ).to(self._device)
            text_emb = F.embedding(text_enc.input_ids, self.embed_weight.half())
            mask = text_enc.attention_mask.unsqueeze(-1).float()
            lang_emb = (text_emb.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            self.lang_holder[0] = self.film_gen(lang_emb)

        try:
            action_np = self.model.predict_action(
                **inputs, unnorm_key=UNNORM_KEY, do_sample=False
            )
        finally:
            self.lang_holder[0] = None
        return action_np.tolist()


# ---------------------------------------------------------------------------
#  TCP server
# ---------------------------------------------------------------------------

def run_server(args):
    import socket

    server = OpenVLAServer(lora_dir=args.lora_path, model_dir=args.model_dir)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    print(f"[OpenVLAServer] TCP listening on {args.host}:{args.port}", flush=True)

    n_reqs, total_ms = 0, 0.0

    def handle(conn, addr):
        nonlocal n_reqs, total_ms
        print(f"[OpenVLAServer] Client from {addr}", flush=True)
        with conn:
            while True:
                raw = b""
                while len(raw) < 4:
                    c = conn.recv(4 - len(raw))
                    if not c:
                        return
                    raw += c
                req_len = int.from_bytes(raw, "big")
                chunks, got = [], 0
                while got < req_len:
                    c = conn.recv(min(65536, req_len - got))
                    if not c:
                        return
                    chunks.append(c)
                    got += len(c)

                data = json.loads(b"".join(chunks))
                seq  = data.get("seq", 0)

                try:
                    obs_rgb = np.array(
                        Image.open(io.BytesIO(base64.b64decode(data["jpeg_b64"]))).convert("RGB"),
                        dtype=np.uint8,
                    )
                except Exception as e:
                    _send(conn, {"seq": seq, "action": [0.0]*7,
                                 "status": "error", "message": str(e)})
                    continue

                instr = data.get("instruction", args.instruction)

                if n_reqs == 0:
                    print(f"[DEBUG] image shape={obs_rgb.shape} "
                          f"min={obs_rgb.min()} max={obs_rgb.max()} "
                          f"mean={obs_rgb.mean():.1f}", flush=True)
                    print(f"[DEBUG] instruction: {instr!r}", flush=True)

                t0     = time.perf_counter()
                action = server.predict(obs_rgb, instr)
                ms     = (time.perf_counter() - t0) * 1000

                if n_reqs < 3:
                    print(f"[DEBUG] action = [{', '.join(f'{v:+.4f}' for v in action)}]",
                          flush=True)

                n_reqs   += 1
                total_ms += ms

                _send(conn, {
                    "seq":          seq,
                    "action":       action,    # flat list of 7 floats; runner wraps to (1,7)
                    "exec_horizon": 1,
                    "status":       "ok",
                })

                if n_reqs % 20 == 0:
                    print(f"[OpenVLAServer] {n_reqs} reqs  "
                          f"avg={total_ms/n_reqs:.1f} ms", flush=True)

    try:
        while True:
            conn, addr = sock.accept()
            handle(conn, addr)
    except KeyboardInterrupt:
        print(f"\n[OpenVLAServer] Done after {n_reqs} requests.")
    finally:
        sock.close()


def _send(conn, obj: dict):
    payload = json.dumps(obj).encode()
    conn.sendall(len(payload).to_bytes(4, "big") + payload)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenVLA discrete-token: fine-tune (CE on action tokens) and serve"
    )

    p.add_argument("--serve", action="store_true",
                   help="Run TCP inference server instead of fine-tuning")

    # Model
    p.add_argument("--model_dir", type=str, default="openvla/openvla-7b",
                   help="Base OpenVLA model (HuggingFace ID or local path)")

    # Fine-tuning
    p.add_argument("--resume_from", type=str, default=None,
                   help="checkpoint_epochN dir to resume from")
    p.add_argument("--reset_optimizer", action="store_true",
                   help="Resume weights only; discard saved optimizer state")
    p.add_argument("--finetune_output", type=str, default="outputs/openvla_discrete",
                   help="Output directory for checkpoints")

    # Data
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="Task names or 'all'. Reads data/<task>/demos.hdf5 + task.json.")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--episodes_per_task", type=int, default=25,
                   help="Episodes sampled per task per epoch")
    p.add_argument("--dagger_episodes_per_task", type=int, default=0,
                   help="Also sample this many episodes per task from "
                        "data/<task>/dagger.hdf5 each epoch (0=off). Aggregates "
                        "on-policy corrections WITH the expert demos; never "
                        "train on dagger.hdf5 alone.")
    p.add_argument("--frame_stride", type=int, default=1)

    # Training hyper-params
    p.add_argument("--lr_decay", action="store_true",
                   help="Cosine-decay the lr over the (resumed) epoch span")
    p.add_argument("--lr_min_ratio", type=float, default=0.02,
                   help="eta_min = lr * this (cosine decay floor)")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=2,
                   help="Samples per gradient step (processed serially for device_map=auto)")
    p.add_argument("--lr",         type=float, default=5e-4)

    # LoRA
    p.add_argument("--lora_rank",  type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)

    # FiLM language conditioning (strengthens color grounding)
    p.add_argument("--use_film", action="store_true",
                   help="Insert FiLM blocks conditioned on the instruction")
    p.add_argument("--film_hidden", type=int, default=256)

    # Gripper-token loss weighting (sharpens close-timing precision)
    p.add_argument("--gripper_loss_weight", type=float, default=1.0,
                   help="Weight on the gripper token (dim 6) in the CE loss; >1 sharpens timing")

    # Gripper-transition oversampling
    p.add_argument("--non_transition_prob", type=float, default=0.0,
                   help="Keep non-transition frames with this prob (0=off). 0.1 ≈ 10x upweight")
    p.add_argument("--transition_lookahead", type=int, default=16,
                   help="Frames ahead to scan for a gripper flip")

    # Inference server
    p.add_argument("--lora_path", type=str,
                   default="outputs/openvla_finetuned_v3/checkpoint_epoch6",
                   help="Dir containing lora_weights.pt OR adapter_model.safetensors")
    p.add_argument("--instruction", type=str,
                   default="pick up the red cube and place it on the tray",
                   help="Fallback instruction when client does not send one")
    p.add_argument("--exec_horizon", type=int, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5557,
                   help="TCP port (default 5557 — avoids clash with OFT on 5556)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.serve:
        run_server(args)
    else:
        finetune(args)
