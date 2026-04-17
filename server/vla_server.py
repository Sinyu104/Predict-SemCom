"""
vla_server.py  —  OpenVLA inference server (runs on the Linux server).

Runs on: Linux server with 4 × T4 GPUs
Transport: ROS2 pub/sub via Fast DDS Discovery Server

Message format
--------------
Client → Server:  /vla/request  (std_msgs/String)
    JSON: {
        "seq":         <int>,          sequence number for request/reply matching
        "jpeg_b64":    "<base64>",     JPEG-compressed RGB frame
        "instruction": "pick up ...",
        "unnorm_key":  "franka_isaac"
    }

Server → Client:  /vla/response (std_msgs/String)
    JSON: {"seq": <int>, "action": [a0, a1, a2, a3, a4, a5, a6], "status": "ok"}
    On error: {"seq": <int>, "action": [0,0,0,0,0,0,0], "status": "error", "message": "..."}

ROS2 environment (set on BOTH machines before launching)
---------------------------------------------------------
    export ROS_DOMAIN_ID=66
    export ROS_DISCOVERY_SERVER=10.32.33.41:11811

Usage
-----
On the Linux server (run BEFORE starting isaac_collector.py):

    # With fine-tuned weights:
    python server/vla_server.py --model_dir outputs/openvla_finetuned

    # With base weights (zero-shot):
    python server/vla_server.py \
        --model_dir openvla/openvla-7b \
        --unnorm_key bridge_orig

    # Fine-tuning:
    python server/vla_server.py --mode finetune \
        --demo_data data/franka_demos.hdf5 \
        --model_dir openvla/openvla-7b \
        --finetune_output outputs/openvla_finetuned

Dependencies
------------
    pip install numpy pillow transformers timm tokenizers peft accelerate
    # ROS2: source /opt/ros/<distro>/setup.bash  (provides rclpy)
"""

import argparse
import base64
import io
import json
import os
import queue
import sys
import threading
import time

import numpy as np
from PIL import Image


# ========================================================================== #
#  OpenVLA model wrapper                                                      #
# ========================================================================== #

class OpenVLAServer:
    """
    Loads OpenVLA onto all available GPUs using HuggingFace device_map="auto".
    With 4 × T4 (16 GB each = 64 GB total), the 7B model (≈14 GB float16)
    fits easily with room for activations.

    Parameters
    ----------
    model_dir   : str  HuggingFace model ID or path to fine-tuned weights
    unnorm_key  : str  Action un-normalisation key
    device_map  : str  "auto" distributes layers across all available GPUs
    quantize    : bool 4-bit quantisation (use if VRAM is tight)
    """

    def __init__(
        self,
        model_dir:  str,
        unnorm_key: str  = "franka_isaac",
        device_map: str  = "auto",
        quantize:   bool = False,
    ):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch
        except ImportError:
            raise ImportError(
                "Install transformers: pip install transformers timm tokenizers"
            )

        self.unnorm_key = unnorm_key
        self.device_map = device_map

        print(f"[VLAServer] Loading model from '{model_dir}' …")
        print(f"[VLAServer] device_map='{device_map}'  quantize={quantize}")

        load_kwargs = dict(
            device_map       = device_map,
            low_cpu_mem_usage= True,
            trust_remote_code= True,
        )
        if quantize:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=__import__("torch").float16,
                )
            except ImportError:
                print("[VLAServer] bitsandbytes not found, using float16")
                load_kwargs["torch_dtype"] = __import__("torch").float16
        else:
            load_kwargs["torch_dtype"] = __import__("torch").float16

        # Detect PEFT adapter: adapter_config.json tells us the base model
        adapter_config_path = os.path.join(model_dir, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            from peft import PeftModel
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg["base_model_name_or_path"]
            print(f"[VLAServer] PEFT adapter detected. Base model: '{base_model_id}'")

            self.processor = AutoProcessor.from_pretrained(
                base_model_id, trust_remote_code=True
            )
            base = AutoModelForVision2Seq.from_pretrained(base_model_id, **load_kwargs)
            peft_model = PeftModel.from_pretrained(base, model_dir)
            # Merge LoRA weights into the base model and discard the adapter
            # structure.  This eliminates device mismatches that arise when
            # device_map="auto" places base-layer tensors on one GPU and the
            # separately-loaded LoRA A/B matrices on another.
            print("[VLAServer] Merging LoRA weights into base model …")
            self.model = peft_model.merge_and_unload()
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_dir, trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_dir, **load_kwargs
            )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Merge custom dataset_statistics.json (written during fine-tuning) into
        # the model's norm_stats so that custom unnorm_keys (e.g. "franka_isaac")
        # are recognised.  This is necessary when loading a PEFT adapter because
        # the base model only knows about its original training datasets.
        stats_path = os.path.join(model_dir, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                extra_stats = json.load(f)
            if hasattr(self.model, "norm_stats"):
                self.model.norm_stats.update(extra_stats)
                print(f"[VLAServer] Loaded norm_stats from {stats_path}")
            else:
                print(f"[VLAServer] WARNING: model has no norm_stats attribute; "
                      f"cannot inject custom dataset statistics.")

        # Log which device each major component landed on
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"[VLAServer] {n_gpus} GPU(s) available:")
            for i in range(n_gpus):
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
        # Build action tokenizer using the same processor.tokenizer used during
        # fine-tuning so that token ID ↔ bin mapping is identical at inference.
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self._debug_first_call = True   # print full debug on the first request only

        # ── Checkpoint verification ───────────────────────────────────── #
        # 1. Confirm which file(s) were actually loaded
        print(f"[VLAServer] model_dir resolved to: '{model_dir}'")
        if os.path.isfile(os.path.join(model_dir, "adapter_config.json")):
            # List LoRA weight files so we can confirm they are non-empty
            lora_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(".bin") or f.endswith(".safetensors")
            ]
            total_bytes = sum(
                os.path.getsize(os.path.join(model_dir, f)) for f in lora_files
            )
            print(f"[VLAServer] LoRA adapter files : {lora_files}")
            print(f"[VLAServer] LoRA total size    : {total_bytes / 1e6:.1f} MB")
        else:
            cfg_path = os.path.join(model_dir, "config.json")
            print(f"[VLAServer] Full model config  : {cfg_path}  "
                  f"(exists={os.path.isfile(cfg_path)})")

        # 2. Show action token ID range — must match what finetune used
        at = self.action_tokenizer
        print(f"[VLAServer] ActionTokenizer: vocab={len(self.processor.tokenizer)} "
              f"| action_token_ids[0]={at.action_token_ids[0]} "
              f"… [{at.n_bins-1}]={at.action_token_ids[-1]}")

        # 3. Round-trip sanity check: encode a known action, decode it back
        _test_action = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _test_ids    = at(_test_action)
        _token_to_bin = {int(tid): i for i, tid in enumerate(at.action_token_ids)}
        _bin_idx  = np.array([_token_to_bin[tid] for tid in _test_ids])
        _centers  = (at.bin_edges[:-1] + at.bin_edges[1:]) / 2.0
        _decoded  = _centers[_bin_idx]
        print(f"[VLAServer] Round-trip check:")
        print(f"  original : {_test_action.tolist()}")
        print(f"  token_ids: {_test_ids}")
        print(f"  decoded  : {_decoded.tolist()}")
        print("[VLAServer] Model loaded and frozen.")

    @__import__("torch").inference_mode()
    def predict(
        self,
        obs_rgb:     np.ndarray,   # (H, W, 3) uint8
        instruction: str,
    ) -> list:
        """
        Run OpenVLA inference on a single RGB frame.

        Uses the same ActionTokenizer that was used during fine-tuning to
        decode generated token IDs back to continuous actions, guaranteeing
        that the token ID ↔ bin mapping is identical between training and
        inference.  Calling model.predict_action() instead would use OpenVLA's
        internal decoder, which may differ from our custom ActionTokenizer.

        Returns
        -------
        list of 7 floats: [x, y, z, rx, ry, rz, gripper]
        """
        import torch

        pil_img = Image.fromarray(obs_rgb)
        prompt  = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self.processor(prompt, pil_img, return_tensors="pt").to(
            "cuda:0",
            dtype=torch.float16,
        )

        # Generate exactly 7 action tokens
        generated = self.model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
        )

        # The last 7 token IDs in the output are the action tokens
        action_token_ids = generated[0, -7:].cpu().numpy()

        # Map token IDs → bin indices (inverse of ActionTokenizer.__call__)
        token_to_bin = {
            int(tid): i
            for i, tid in enumerate(self.action_tokenizer.action_token_ids)
        }
        unknown_ids = [int(tid) for tid in action_token_ids if int(tid) not in token_to_bin]
        bin_indices = np.array([
            token_to_bin.get(int(tid), self.action_tokenizer.n_bins // 2)
            for tid in action_token_ids
        ])

        # Map bin indices → continuous values via bin centres
        bin_centers = (
            self.action_tokenizer.bin_edges[:-1]
            + self.action_tokenizer.bin_edges[1:]
        ) / 2.0
        actions = bin_centers[bin_indices]

        # ── First-call debug dump ─────────────────────────────────────── #
        if self._debug_first_call:
            self._debug_first_call = False
            print("\n[VLAServer] ── FIRST INFERENCE DEBUG ──────────────────────")
            print(f"  prompt              : {prompt!r}")
            print(f"  input_ids shape     : {inputs['input_ids'].shape}")
            print(f"Min: {inputs['input_ids'].min().item()}")
            print(f"Max: {inputs['input_ids'].max().item()}")
            print(f"Avg: {inputs['input_ids'].float().mean().item():.2f}")
            print(f"  generated shape     : {generated.shape}")
            print(f"  raw generated ids   : {generated[0].tolist()}")
            print(f"  action token ids    : {action_token_ids.tolist()}")
            print(f"  valid action range  : [{self.action_tokenizer.action_token_ids[0]}"
                  f" … {self.action_tokenizer.action_token_ids[-1]}]")
            if unknown_ids:
                print(f"  WARNING unknown ids : {unknown_ids}  "
                      f"(model generated non-action tokens — mapped to bin 128 / value ~0)")
            else:
                print(f"  all token ids are valid action tokens")
            print(f"  bin_indices         : {bin_indices.tolist()}")
            print(f"  decoded actions     : {actions.tolist()}")
            print("[VLAServer] ────────────────────────────────────────────────\n")

        return actions.tolist()   # list of 7 Python floats


# ========================================================================== #
#  Action tokenizer                                                           #
# ========================================================================== #

class ActionTokenizer:
    """
    Maps continuous 7-DoF actions to LLaMA token IDs following OpenVLA's convention.

    OpenVLA repurposes the 256 least-used tokens in LLaMA-2's vocabulary as
    action bins.  In LLaMA-2 the vocabulary is roughly ordered by corpus
    frequency, so the 256 tokens with the *highest* IDs are the least frequent
    and are safe to reuse as discrete action tokens without conflicting with
    natural-language generation.

    Using raw bin indices 0–255 as token IDs (the previous fallback) is wrong:
    those IDs correspond to extremely common tokens (basic ASCII / punctuation),
    so the model's pre-trained weights actively resist predicting them in the
    action-output position.
    """

    def __init__(self, tokenizer, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0):
        import numpy as _np
        self.tokenizer   = tokenizer
        self.n_bins      = bins
        self.min_action  = min_action
        self.max_action  = max_action

        # Uniform discretisation grid over [-1, 1]
        self.bin_edges   = _np.linspace(min_action, max_action, bins + 1)

        # Identify the 256 least-used token IDs:
        # sort all non-special vocab IDs and take the highest `bins` of them.
        special_ids = set(tokenizer.all_special_ids)
        all_ids     = sorted(
            [tid for tid in tokenizer.get_vocab().values() if tid not in special_ids]
        )
        self.action_token_ids = _np.array(all_ids[-bins:])   # highest IDs = least frequent

    def __call__(self, action) -> list:
        """
        Discretise a (7,) float action array and return a list of 7 token IDs.
        """
        import numpy as _np
        action      = _np.clip(action, self.min_action, self.max_action)
        bin_indices = _np.digitize(action, self.bin_edges[1:-1])   # 0 … bins-1
        bin_indices = _np.clip(bin_indices, 0, self.n_bins - 1)
        return self.action_token_ids[bin_indices].tolist()


# ========================================================================== #
#  Stage 0: Fine-tune OpenVLA on the server                                  #
# ========================================================================== #

def finetune_openvla(args):
    """
    Fine-tune base OpenVLA on Franka demonstrations using LoRA.

    The demo HDF5 file must be accessible on the server (SCP it over
    from the desktop after collection).

    Output: args.finetune_output/  (LoRA adapter weights)
    """
    import torch
    import h5py
    from torch.optim import AdamW
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from transformers import AutoModelForVision2Seq, AutoProcessor

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("Install peft: pip install peft accelerate")

    # ── Resolve start epoch when resuming ────────────────────────────── #
    import re
    start_epoch = 0
    if args.resume_from:
        basename = os.path.basename(args.resume_from.rstrip("/"))
        m = re.match(r"checkpoint_epoch(\d+)$", basename)
        if m:
            start_epoch = int(m.group(1))
        else:
            raise ValueError(
                f"--resume_from path must end with 'checkpoint_epochN', "
                f"got: {args.resume_from!r}"
            )
        print(f"\n[finetune] Resuming from  : {args.resume_from}  (epoch {start_epoch})")
    else:
        print(f"\n[finetune] Base model : {args.model_dir}")

    print(f"[finetune] Demo data  : {args.demo_data}")
    print(f"[finetune] Output     : {args.finetune_output}")

    if args.resume_from:
        # Load base model, then restore the LoRA adapter from the checkpoint.
        from peft import PeftModel
        from peft import LoraConfig as _LC  # noqa — keep import happy
        with open(os.path.join(args.resume_from, "adapter_config.json")) as _f:
            _adapter_cfg = json.load(_f)
        _base_id = _adapter_cfg["base_model_name_or_path"]
        print(f"[finetune] Base model (from adapter config): {_base_id}")
        processor = AutoProcessor.from_pretrained(_base_id, trust_remote_code=True)
        _base = AutoModelForVision2Seq.from_pretrained(
            _base_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(_base, args.resume_from, is_trainable=True)
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_dir, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )

        # Apply LoRA — only the language backbone attention layers are tuned.
        # Vision encoder weights are left frozen for stability.
        lora_cfg = LoraConfig(
            r             = args.lora_rank,
            lora_alpha    = args.lora_rank * 2,
            target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout  = 0.05,
            bias          = "none",
            task_type     = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    print(f"[finetune] Action token IDs: {action_tokenizer.action_token_ids[0]} … {action_tokenizer.action_token_ids[-1]}")

    n_gpus = torch.cuda.device_count()
    print(f"[finetune] Using {n_gpus} GPU(s) — model sharded via device_map='auto'")

    # ── Demo dataset ─────────────────────────────────────────────────── #
    class DemoDataset(TorchDataset):
        def __init__(self, path: str, frame_stride: int = 1):
            self.samples = []
            n_eps = 0
            with h5py.File(path, "r") as f:
                for ep_key in sorted(f.keys()):
                    grp = f[ep_key]
                    if "observations" not in grp or "actions" not in grp:
                        continue
                    obs  = np.array(grp["observations"])   # (T, H, W, 3)
                    acts = np.array(grp["actions"])         # (T, 7)
                    for t in range(0, len(obs) - 1, frame_stride):
                        self.samples.append((obs[t], acts[t].astype(np.float32)))
                    n_eps += 1
            print(f"[finetune] {n_eps} episodes → {len(self.samples)} samples "
                  f"(stride={frame_stride})")

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = DemoDataset(args.demo_data, frame_stride=args.frame_stride)
    loader  = DataLoader(
        dataset, batch_size=args.finetune_batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    print(f"[finetune] {len(dataset)} demonstration steps")

    total_epochs = start_epoch + args.finetune_epochs

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.finetune_lr,
    )
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # Restore optimizer / scheduler state when resuming
    if args.resume_from:
        train_state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.isfile(train_state_path):
            state = torch.load(train_state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            print(f"[finetune] Restored optimizer + scheduler from {train_state_path}")
        else:
            print(f"[finetune] No training_state.pt found in {args.resume_from}; "
                  f"optimizer/scheduler start fresh (LR may differ slightly).")

    from tqdm import tqdm

    prompt_tpl = (
        f"In: What action should the robot take to {args.instruction}?\nOut:"
    )
    model.train()

    epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Epochs", unit="epoch")
    for ep in epoch_bar:
        total_loss = 0.0
        n_steps    = 0

        step_bar = tqdm(loader, desc=f"  Epoch {ep+1}/{args.finetune_epochs}",
                        unit="batch", leave=False)
        for obs_batch, act_batch in step_bar:
            # obs_batch: (B, H, W, 3) uint8
            B = obs_batch.shape[0]

            # Build all PIL images for the batch
            pil_imgs = []
            for i in range(B):
                pil_imgs.append(Image.fromarray(obs_batch[i].numpy()))

            # Single batched processor call instead of B individual calls
            inputs = processor(
                [prompt_tpl] * B, pil_imgs, return_tensors="pt", padding=True
            ).to("cuda:0", dtype=torch.float16)

            # Encode ground-truth actions as token IDs — (B, 7)
            act_np     = act_batch.numpy()                        # (B, 7)
            token_ids  = np.stack([action_tokenizer(act_np[i]) for i in range(B)])
            action_ids = torch.tensor(token_ids, dtype=torch.long, device="cuda:0")

            # Append action tokens to input_ids; mask prompt positions with -100
            prompt_len     = inputs["input_ids"].shape[1]
            action_len     = action_ids.shape[1]                  # always 7
            full_input_ids = torch.cat(
                [inputs["input_ids"], action_ids], dim=1
            )
            labels = torch.cat([
                torch.full((B, prompt_len), -100, dtype=torch.long, device="cuda:0"),
                action_ids,
            ], dim=1)
            full_attn_mask = torch.cat([
                inputs["attention_mask"],
                torch.ones(B, action_len, dtype=inputs["attention_mask"].dtype,
                           device="cuda:0"),
            ], dim=1)

            forward_inputs = {
                k: v for k, v in inputs.items()
                if k not in ("input_ids", "attention_mask")
            }
            forward_inputs["input_ids"]      = full_input_ids
            forward_inputs["attention_mask"] = full_attn_mask

            # Single forward pass for the whole batch
            outputs    = model(**forward_inputs, labels=labels)
            batch_loss = outputs.loss   # already averaged over B by the model

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()

            total_loss += batch_loss.item()
            n_steps    += 1
            step_bar.set_postfix(loss=f"{batch_loss.item():.4f}")

        avg = total_loss / max(n_steps, 1)
        scheduler.step()
        epoch_bar.set_postfix(avg_loss=f"{avg:.4f}")
        print(f"[finetune] Epoch {ep+1}/{total_epochs}  loss={avg:.4f}")

        # Save checkpoint for this epoch, then delete the previous one
        ckpt_dir = os.path.join(args.finetune_output, f"checkpoint_epoch{ep+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        # Save optimizer + scheduler so training can be resumed from this checkpoint
        torch.save(
            {"optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "epoch": ep + 1},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        print(f"[finetune] Checkpoint saved → {ckpt_dir}")

        if ep > start_epoch:
            prev_ckpt = os.path.join(args.finetune_output, f"checkpoint_epoch{ep}")
            if os.path.isdir(prev_ckpt):
                import shutil
                shutil.rmtree(prev_ckpt)
                print(f"[finetune] Removed previous checkpoint: {prev_ckpt}")

    # Copy the final checkpoint to the output root for easy loading
    os.makedirs(args.finetune_output, exist_ok=True)
    final_ckpt = os.path.join(args.finetune_output, f"checkpoint_epoch{total_epochs}")
    model.save_pretrained(args.finetune_output)
    processor.save_pretrained(args.finetune_output)

    # Write action normalisation statistics for the custom unnorm_key
    stats = {
        args.unnorm_key: {
            "action": {
                "mean": [0.0] * 7, "std":  [1.0] * 7,
                "max":  [1.0] * 7, "min":  [-1.0] * 7,
                "q01":  [-1.0] * 7, "q99": [1.0] * 7,
            }
        }
    }
    stats_path = os.path.join(args.finetune_output, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[finetune] Saved to {args.finetune_output}")
    print(f"[finetune] Start the ROS2 server with:")
    print(f"  python server/vla_server.py --model_dir {args.finetune_output}")


# ========================================================================== #
#  Helpers                                                                    #
# ========================================================================== #

def _resolve_model_dir(model_dir: str) -> str:
    """
    Return the directory that actually contains a loadable model.

    If model_dir contains a config.json it is used as-is (covers both
    HuggingFace cache IDs and a fully-written finetune root).

    Otherwise, look for checkpoint_epoch* subdirectories and return the
    one with the highest epoch number.  This handles the common case where
    training was interrupted before the final root-level save.
    """
    def _is_valid(path: str) -> bool:
        """True if path looks like a loadable model (full or PEFT adapter)."""
        return (
            os.path.isfile(os.path.join(path, "adapter_config.json")) or
            os.path.isfile(os.path.join(path, "config.json"))
        )

    if _is_valid(model_dir):
        return model_dir

    # Scan for checkpoint_epoch* subdirectories
    if os.path.isdir(model_dir):
        ckpts = sorted(
            [
                d for d in os.listdir(model_dir)
                if d.startswith("checkpoint_epoch") and
                _is_valid(os.path.join(model_dir, d))
            ],
            key=lambda d: int(d.replace("checkpoint_epoch", "") or 0),
        )
        if ckpts:
            resolved = os.path.join(model_dir, ckpts[-1])
            print(
                f"[VLAServer] No model found at '{model_dir}'. "
                f"Using latest checkpoint: {resolved}"
            )
            return resolved

    # Fall through: HuggingFace model ID or network download
    return model_dir


# ========================================================================== #
#  ROS2 inference server                                                      #
# ========================================================================== #

def run_server(args):
    """
    Start the ROS2 inference server.  Blocks until interrupted with Ctrl-C.

    Subscribes to /vla/request, runs OpenVLA inference, publishes to /vla/response.
    Requests are queued so the ROS2 executor stays responsive while inference runs.

    Required environment variables (set before launching):
        export ROS_DOMAIN_ID=66
        export ROS_DISCOVERY_SERVER=10.32.33.41:11811
    """
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError:
        raise ImportError(
            "rclpy not found. Source your ROS2 workspace:\n"
            "  source /opt/ros/<distro>/setup.bash"
        )

    model_dir = _resolve_model_dir(args.model_dir)
    vla = OpenVLAServer(
        model_dir  = model_dir,
        unnorm_key = args.unnorm_key,
        device_map = args.device_map,
        quantize   = args.quantize,
    )

    rclpy.init()
    node = rclpy.create_node("vla_server")
    pub  = node.create_publisher(String, "/vla/response", 10)

    request_queue = queue.Queue()

    def on_request(msg: String):
        try:
            data = json.loads(msg.data)
            request_queue.put(data)
        except json.JSONDecodeError as e:
            node.get_logger().error(f"Malformed request JSON: {e}")

    node.create_subscription(String, "/vla/request", on_request, 10)

    # Spin executor in a background thread so the main thread can run inference
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("\n[VLAServer] ROS2 node started.")
    print("[VLAServer] Subscribed to /vla/request")
    print("[VLAServer] Publishing to  /vla/response")
    print("[VLAServer] Waiting for Isaac Sim client …\n")

    n_requests = 0
    total_ms   = 0.0

    try:
        while True:
            # Block until a request arrives
            data = request_queue.get()
            seq  = data.get("seq", 0)

            # ── Decode image ─────────────────────────────────────────── #
            try:
                jpeg_bytes = base64.b64decode(data["jpeg_b64"])
                pil_img    = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                obs_rgb    = np.array(pil_img, dtype=np.uint8)
            except Exception as e:
                reply     = String()
                reply.data = json.dumps({
                    "seq": seq, "action": [0.0] * 7,
                    "status": "error", "message": f"Image decode error: {e}",
                })
                pub.publish(reply)
                continue

            instruction = data.get("instruction", args.instruction)

            # ── Inference ────────────────────────────────────────────── #
            t0     = time.perf_counter()
            action = vla.predict(obs_rgb, instruction)
            ms     = (time.perf_counter() - t0) * 1000

            n_requests += 1
            total_ms   += ms

            # ── Publish response ─────────────────────────────────────── #
            reply      = String()
            reply.data = json.dumps({"seq": seq, "action": action, "status": "ok"})
            pub.publish(reply)

            if n_requests % 50 == 0:
                print(
                    f"[VLAServer] {n_requests} requests  "
                    f"avg_latency={total_ms/n_requests:.1f} ms"
                )

    except KeyboardInterrupt:
        print(f"\n[VLAServer] Shutting down after {n_requests} requests.")
    finally:
        rclpy.shutdown()


# ========================================================================== #
#  CLI                                                                        #
# ========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenVLA ROS2 Server — runs on Linux server with 4× T4"
    )
    p.add_argument("--mode", type=str, default="serve",
                   choices=["serve", "finetune"],
                   help="serve: run ROS2 inference server; "
                        "finetune: fine-tune OpenVLA on demo data")

    # Common
    p.add_argument("--model_dir",    type=str,
                   default="outputs/openvla_finetuned",
                   help="HuggingFace model ID or path to fine-tuned weights")
    p.add_argument("--unnorm_key",   type=str, default="franka_isaac")
    p.add_argument("--instruction",  type=str,
                   default="pick up the red cube and place it on the tray")

    # Server
    p.add_argument("--device_map", type=str,  default="auto",
                   help="HuggingFace device_map: 'auto' spreads across all GPUs")
    p.add_argument("--quantize",   action="store_true",
                   help="4-bit quantisation (saves VRAM, slight accuracy drop)")

    # Fine-tuning
    p.add_argument("--demo_data",          type=str,
                   default="data/franka_demos.hdf5")
    p.add_argument("--finetune_output",    type=str,
                   default="outputs/openvla_finetuned")
    p.add_argument("--finetune_epochs",    type=int,   default=10)
    p.add_argument("--finetune_lr",        type=float, default=2e-5)
    p.add_argument("--finetune_batch_size",type=int,   default=4)
    p.add_argument("--lora_rank",          type=int,   default=32)
    p.add_argument("--frame_stride",       type=int,   default=1,
                   help="Use every Nth frame per episode (e.g. 3 = 3x fewer samples, "
                        "3x faster training). Adjacent frames are nearly identical "
                        "so stride=3-5 loses little information.")
    p.add_argument("--resume_from",        type=str,   default=None,
                   help="Path to a checkpoint_epochN directory to resume training "
                        "from (e.g. outputs/openvla_finetuned_v2/checkpoint_epoch6). "
                        "The LoRA adapter and optimizer/scheduler state are restored "
                        "and epoch numbering continues from N.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "finetune":
        finetune_openvla(args)
    else:
        run_server(args)