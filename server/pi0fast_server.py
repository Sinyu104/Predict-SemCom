"""
pi0fast_server.py  —  PI0Fast fine-tuning on Franka demonstrations.

Mirrors the structure of vla_server.py but targets lerobot's PI0FastPolicy
instead of OpenVLA.

HDF5 layout (data/pick_blue_cube_to_tray/demos.hdf5):
    /episode_N/observations_cam1  (T, 224, 224, 3) uint8
    /episode_N/observations_cam2  (T, 224, 224, 3) uint8
    /episode_N/actions            (T, 7)           float32

Usage
-----
Fine-tune (all parameters):
    conda activate worldmodel
    python server/pi0fast_server.py \\
        --model_dir /path/to/pi0-fast \\
        --demo_data data/pick_blue_cube_to_tray/demos.hdf5 \\
        --finetune_output outputs/pi0fast_finetuned \\
        --instruction "pick up the blue cube and place it on the tray" \\
        --epochs 20 --batch_size 4

Fine-tune (action-expert only — less VRAM):
    python server/pi0fast_server.py \\
        --model_dir /path/to/pi0-fast \\
        --demo_data data/pick_blue_cube_to_tray/demos.hdf5 \\
        --finetune_output outputs/pi0fast_finetuned \\
        --freeze_vision --freeze_language

Resume from checkpoint:
    python server/pi0fast_server.py \\
        --resume_from outputs/pi0fast_finetuned/checkpoint_epoch5 \\
        --demo_data data/pick_blue_cube_to_tray/demos.hdf5 \\
        --finetune_output outputs/pi0fast_finetuned

Camera keys in the batch default to:
    observation.images.cam1   ←  observations_cam1
    observation.images.cam2   ←  observations_cam2
Override with --cam1_key / --cam2_key to match the pretrained model's config.
"""

import argparse
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert continuous action chunks to FAST token IDs using the tokenisers
    already loaded inside the policy.

    Replicates ActionTokenizerProcessorStep._tokenize_action without the
    full lerobot processor pipeline.

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

    tokens_list, masks_list = [], []
    for i in range(action_chunks.size(0)):
        act_cpu = action_chunks[i : i + 1].cpu()   # (1, K, action_dim)
        raw     = fast_tok(act_cpu)                 # list or Tensor

        if not isinstance(raw, torch.Tensor):
            raw = torch.tensor(raw, dtype=torch.long)
        if raw.dim() > 1:
            raw = raw.flatten()

        # Map FAST token IDs → PaliGemma vocabulary space
        pali_ids = pali_tok.vocab_size - 1 - fast_skip - raw.long()

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


# ========================================================================== #
#  Fine-tuning                                                                #
# ========================================================================== #

def finetune_pi0fast(args):
    """Fine-tune PI0Fast on dual-camera Franka demo data."""
    _mock_groot()
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    # ── Resolve resume epoch ─────────────────────────────────────────── #
    start_epoch = 0
    if args.resume_from:
        m = re.match(r".*checkpoint_epoch(\d+)$", args.resume_from.rstrip("/"))
        if not m:
            raise ValueError(
                f"--resume_from must point to a checkpoint_epochN directory, "
                f"got: {args.resume_from!r}"
            )
        start_epoch = int(m.group(1))
        print(f"\n[finetune] Resuming from: {args.resume_from} (epoch {start_epoch})")
    else:
        print(f"\n[finetune] Base model : {args.model_dir}")

    print(f"[finetune] Demo data  : {args.demo_data}")
    print(f"[finetune] Output     : {args.finetune_output}")

    # ── Load model ───────────────────────────────────────────────────── #
    model_path = args.resume_from if args.resume_from else args.model_dir
    print(f"[finetune] Loading PI0FastPolicy from {model_path} …")

    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(model_path)
    cfg.chunk_size    = args.chunk_size
    cfg.n_action_steps = args.chunk_size
    print(f"[finetune] chunk_size = {args.chunk_size}")

    policy = PI0FastPolicy.from_pretrained(model_path, config=cfg)
    policy.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    print(f"[finetune] Device: {device}  ({torch.cuda.device_count()} GPU(s))")

    # ── Resolve camera keys ───────────────────────────────────────────── #
    # Use keys from the model config if present, otherwise use the CLI defaults.
    cfg_img_keys = list(getattr(policy.config, "image_features", {}).keys())
    if len(cfg_img_keys) >= 2:
        cam1_key = cfg_img_keys[0]
        cam2_key = cfg_img_keys[1]
        print(f"[finetune] Camera keys from model config: {cam1_key}, {cam2_key}")
    elif len(cfg_img_keys) == 1:
        cam1_key = cfg_img_keys[0]
        cam2_key = args.cam2_key
        print(f"[finetune] Model config has 1 image key ({cam1_key}); "
              f"cam2 → '{cam2_key}' (override with --cam2_key)")
    else:
        cam1_key = args.cam1_key
        cam2_key = args.cam2_key
        print(f"[finetune] No image_features in model config; "
              f"using '{cam1_key}', '{cam2_key}'")

    # CLI overrides always win
    if args.cam1_key:
        cam1_key = args.cam1_key
    if args.cam2_key:
        cam2_key = args.cam2_key
    print(f"[finetune] Batch image keys: '{cam1_key}', '{cam2_key}'")

    # ── Freeze sub-modules ────────────────────────────────────────────── #
    if args.freeze_vision:
        for name, p in policy.named_parameters():
            if "vision" in name or "paligemma" in name.lower():
                p.requires_grad_(False)
        print("[finetune] Vision/PaliGemma backbone frozen.")

    if args.freeze_language:
        for name, p in policy.named_parameters():
            if "language" in name:
                p.requires_grad_(False)
        print("[finetune] Language model frozen.")

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[finetune] Trainable: {n_train:,} / {n_total:,} params")

    # ── Dataset & loader ─────────────────────────────────────────────── #
    dataset = DemoDataset(
        path         = args.demo_data,
        chunk_size   = policy.config.chunk_size,
        frame_stride = args.frame_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,
        drop_last   = True,
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────── #
    total_epochs = start_epoch + args.epochs
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=args.lr * 0.1,
    )

    if args.resume_from:
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            print(f"[finetune] Restored optimizer + scheduler from {state_path}")

    # ── Training loop ─────────────────────────────────────────────────── #
    os.makedirs(args.finetune_output, exist_ok=True)

    for ep in tqdm(range(start_epoch, total_epochs), desc="Epochs", unit="epoch"):
        policy.train()
        total_loss = 0.0
        n_steps    = 0

        step_bar = tqdm(loader, desc=f"  Epoch {ep+1}/{total_epochs}",
                        unit="batch", leave=False)

        for cam1_imgs, cam2_imgs, action_chunks in step_bar:
            # cam1_imgs, cam2_imgs : (B, 3, 224, 224) float [0, 1]
            # action_chunks        : (B, K, 7) float32
            cam1_imgs     = cam1_imgs.to(device)
            cam2_imgs     = cam2_imgs.to(device)
            action_chunks = action_chunks.to(device)
            B             = cam1_imgs.size(0)

            act_tokens, act_masks = tokenize_action_chunks(action_chunks, policy)
            act_tokens = act_tokens.to(device)
            act_masks  = act_masks.to(device)

            lang_ids, lang_masks = build_language_tokens(
                args.instruction, args.state_dim, policy, B, device,
            )

            batch = {
                cam1_key:                              cam1_imgs,
                cam2_key:                              cam2_imgs,
                "observation.language.tokens":         lang_ids,
                "observation.language.attention_mask": lang_masks,
                "action.tokens":                       act_tokens,
                "action.token_mask":                   act_masks,
            }

            loss, loss_dict = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, policy.parameters()), 1.0,
            )
            optimizer.step()

            total_loss += loss.item()
            n_steps    += 1
            step_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 ce=f"{loss_dict.get('ce_loss', 0):.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        scheduler.step()
        print(f"[finetune] Epoch {ep+1}/{total_epochs}  "
              f"loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Checkpoint ───────────────────────────────────────────────── #
        ckpt_dir = os.path.join(args.finetune_output, f"checkpoint_epoch{ep+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        policy.save_pretrained(ckpt_dir)
        torch.save(
            {"optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "epoch":     ep + 1},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        print(f"[finetune] Checkpoint → {ckpt_dir}")

        if ep > start_epoch:
            prev = os.path.join(args.finetune_output, f"checkpoint_epoch{ep}")
            if os.path.isdir(prev):
                shutil.rmtree(prev)

    policy.save_pretrained(args.finetune_output)
    print(f"\n[finetune] Done. Weights → {args.finetune_output}")
    print(f'[finetune] Set in config.py:  "pi0fast_model_path": "{args.finetune_output}"')


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
    p.add_argument("--cam1_key", type=str, default="observation.images.cam1",
                   help="Batch key for camera 1 (observations_cam1)")
    p.add_argument("--cam2_key", type=str, default="observation.images.cam2",
                   help="Batch key for camera 2 (observations_cam2)")

    # Data
    p.add_argument("--demo_data", type=str,
                   default="data/pick_blue_cube_to_tray/demos.hdf5")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Sample every Nth frame per episode")

    # Training
    p.add_argument("--chunk_size",  type=int,   default=10,
                   help="Action chunk length (default 10 matches Stage 1 num_pred)")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2.5e-5)
    p.add_argument("--state_dim",   type=int,   default=6,
                   help="Robot state dim for language prompt (zeros used)")
    p.add_argument("--instruction", type=str,
                   default="pick up the blue cube and place it on the tray")

    # Freeze
    p.add_argument("--freeze_vision",   action="store_true",
                   help="Freeze SigLIP/PaliGemma vision tower")
    p.add_argument("--freeze_language", action="store_true",
                   help="Freeze language model backbone")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune_pi0fast(args)
