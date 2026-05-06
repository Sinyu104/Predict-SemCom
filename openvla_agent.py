"""
openvla_agent.py  —  OpenVLA wrapper for the Predictive SemCom system.

Two classes
-----------
OpenVLAAgent
    Wraps the real 7B model from HuggingFace.  Exposes:
      • encode_image()             — frozen ViT forward, no gradient
      • task_loss_from_tokens()    — gradient flows vit_tokens → projector → LLM
      • predict_action_from_tokens() — greedy action decoding at inference

    All VLA backbone weights are ALWAYS frozen.  Only the SemCom modules
    (TokenEncoder, TokenDecoder, etc.) receive gradients.

OpenVLAStub
    CPU-only placeholder with the same interface as OpenVLAAgent.
    Used for shape-checking and unit tests without GPU or model download.

How task_loss_from_tokens works
--------------------------------
OpenVLA's projector sits between the ViT backbone and the LLM.  To let
gradients flow from the task loss back through the TokenDecoder → latent
space, we inject our reconstructed vit_tokens at the backbone output via a
forward hook.  The hook replaces the backbone output with vit_tokens, so
the projector and LLM see our tokens while the gradient path is preserved:

    loss → logits → LLM → projector → vit_tokens → TokenDecoder → ẑ_t
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# ========================================================================== #
#  Action Tokenizer  (matches vla_server.py / original OpenVLA fine-tuning)  #
# ========================================================================== #

class ActionTokenizer:
    """
    Maps continuous 7-DoF actions to LLaMA token IDs following OpenVLA's convention.

    OpenVLA repurposes the 256 least-used tokens in LLaMA-2's vocabulary as
    action bins — the 256 highest non-special token IDs.
    """
    def __init__(self, tokenizer, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0):
        self.n_bins     = bins
        self.min_action = min_action
        self.max_action = max_action
        self.bin_edges  = np.linspace(min_action, max_action, bins + 1)  # 257 edges → 256 bins
        special_ids     = set(tokenizer.all_special_ids)
        all_ids         = sorted([tid for tid in tokenizer.get_vocab().values() if tid not in special_ids])
        self.action_token_ids = np.array(all_ids[-bins:])                 # highest IDs = least frequent

    def __call__(self, action) -> list:
        action      = np.clip(action, self.min_action, self.max_action)
        bin_indices = np.digitize(action, self.bin_edges[1:-1])           # 255 inner edges → 0…255
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return self.action_token_ids[bin_indices].tolist()


# ========================================================================== #
#  Real OpenVLA Agent                                                         #
# ========================================================================== #

class OpenVLAAgent(nn.Module):
    """
    Frozen OpenVLA-7B wrapper with token-level gradient injection.

    Parameters
    ----------
    instruction : str    Language instruction, e.g. "pick up the red cube"
    unnorm_key  : str    Dataset key for action un-normalisation.
    model_name  : str    HuggingFace model ID
    device      : str    Device for VLA weights (e.g. "cuda:0" or "auto")
    quantize    : bool   4-bit loading via bitsandbytes
    """

    ACTION_TOKEN_COUNT = 7

    def __init__(
        self,
        instruction: str  = "pick up the red cube and place it on the tray",
        unnorm_key:  str  = "bridge_orig",
        model_name:  str  = "openvla/openvla-7b",
        device:      str  = "cuda:0",
        quantize:    bool = False,
    ):
        super().__init__()
        self.instruction = instruction
        self.unnorm_key  = unnorm_key
        self._model_name = model_name
        self._vla_device = device
        self._quantize   = quantize

        self._processor  = None
        self._vla        = None
        self._loaded     = False

        self.register_buffer("_dummy", torch.zeros(1))

    # ------------------------------------------------------------------ #
    #  Lazy loader                                                         #
    # ------------------------------------------------------------------ #

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "Install transformers>=4.40.0: "
                "pip install transformers timm tokenizers"
            )
        print(f"[OpenVLAAgent] Loading {self._model_name} on {self._vla_device}…")
        self._processor = AutoProcessor.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=True,
                      attn_implementation="eager")
        if self._quantize:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16

        if self._vla_device == "auto":
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                # Load to CPU as plain tensors — no device_map, no accelerate
                # hooks.  Using device_map={"":"cpu"} installs CPU AlignDeviceHooks
                # that conflict with the subsequent dispatch_model call in
                # add_lora().  Plain loading avoids this entirely.
                # Dispatch to GPUs happens at the end of add_lora() once all
                # LoRA weights are real tensors on CPU.
                self._vla = AutoModelForVision2Seq.from_pretrained(
                    self._model_name,
                    trust_remote_code      = True,
                    torch_dtype            = torch.float16,
                    attn_implementation    = "eager",
                )
                # _vla_device stays "auto" as dispatch-pending sentinel
            else:
                kwargs["device_map"] = "auto"
                self._vla = AutoModelForVision2Seq.from_pretrained(
                    self._model_name, **kwargs
                )
                self._vla_device = next(self._vla.parameters()).device
        else:
            self._vla = AutoModelForVision2Seq.from_pretrained(
                self._model_name, **kwargs
            ).to(self._vla_device)

        for p in self._vla.parameters():
            p.requires_grad_(False)
        self._vla.eval()

        # Inject dataset_statistics.json if present alongside the checkpoint.
        # Fine-tuned checkpoints save stats here but don't embed them in norm_stats.
        import json, pathlib
        stats_path = pathlib.Path(self._model_name) / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path) as f:
                extra = json.load(f)
            if not hasattr(self._vla, "norm_stats") or self._vla.norm_stats is None:
                self._vla.norm_stats = {}
            self._vla.norm_stats.update(extra)
            print(f"[OpenVLAAgent] Loaded norm_stats keys: {list(extra.keys())}")

        self._action_tokenizer = ActionTokenizer(self._processor.tokenizer)
        print(f"[OpenVLAAgent] ActionTokenizer: IDs {self._action_tokenizer.action_token_ids[0]}"
              f" … {self._action_tokenizer.action_token_ids[-1]}")
        self._loaded = True
        print("[OpenVLAAgent] Loaded and frozen.")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_pil(t: torch.Tensor) -> Image.Image:
        """(3,H,W) float32 [0,1] → PIL RGB image."""
        arr = (t.detach().cpu().clamp(0, 1) * 255).byte()
        return Image.fromarray(arr.permute(1, 2, 0).numpy(), "RGB")

    def _action_to_tokens(self, action_np: np.ndarray, processor=None) -> torch.Tensor:
        """Encode ground-truth (7,) action into OpenVLA token IDs."""
        return torch.tensor(self._action_tokenizer(action_np), dtype=torch.long)

    def _make_prompt(self, instruction: str) -> str:
        return (
            f"In: What action should the robot take to {instruction}?\nOut:"
        )

    def _preprocess_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) float32 [0,1] → (B,6,224,224) float16 for VLA backbone."""
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std
        # Prismatic backbone (DINOv2 + SigLIP) expects 6 channels
        pv = torch.cat([x_norm, x_norm], dim=1)
        return pv.to(self._vla_device, dtype=torch.float16)

    # ------------------------------------------------------------------ #
    #  encode_image  (frozen ViT, no gradient)                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract ViT patch tokens from raw image observations.

        Parameters
        ----------
        x : (B, C, H, W) float32 [0,1]

        Returns
        -------
        tokens : (B, N, D_vit)  float32, detached, on same device as x
        """
        self._load()
        pv     = self._preprocess_pixels(x)
        tokens = self._vla.vision_backbone(pv)
        # vision_backbone may return a tuple; extract the patch token tensor
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        # If backbone returns CLS token prepended, drop it
        if tokens.dim() == 3 and tokens.size(1) == 257:
            tokens = tokens[:, 1:, :]
        return tokens.to(x.device, dtype=x.dtype).detach()

    # ------------------------------------------------------------------ #
    #  task_loss_from_tokens  (gradient flows through vit_tokens)         #
    # ------------------------------------------------------------------ #

    def task_loss_from_tokens(
        self,
        vit_tokens:  torch.Tensor,   # (B, N, D_vit) — from TokenDecoder, with grad
        instructions: str | list,
        actions_gt:  torch.Tensor,   # (B, 7) float32
    ) -> torch.Tensor:
        """
        Differentiable task loss with gradient flowing through vit_tokens.

        Uses a forward hook on vision_backbone to inject vit_tokens as the
        backbone output, bypassing actual image processing.  Gradient path:
            loss → logits → LLM → projector → vit_tokens → TokenDecoder → ẑ_t

        Parameters
        ----------
        vit_tokens   : (B, N, D_vit) with requires_grad (output of TokenDecoder)
        instructions : str or list[str] of length B
        actions_gt   : (B, 7) ground-truth actions

        Returns
        -------
        loss : scalar cross-entropy, differentiable w.r.t. vit_tokens
        """
        self._load()
        B = vit_tokens.size(0)

        if isinstance(instructions, str):
            prompts = [self._make_prompt(instructions)] * B
        else:
            prompts = [self._make_prompt(inst) for inst in instructions]

        # Dummy PIL images for the processor (content irrelevant — hook overrides)
        pil_dummy = [Image.new("RGB", (224, 224)) for _ in range(B)]
        inputs    = self._processor(
            prompts, pil_dummy, return_tensors="pt", padding=True
        )
        prompt_len     = inputs["input_ids"].shape[1]
        input_ids      = inputs["input_ids"].to(self._vla_device)
        attention_mask = inputs["attention_mask"].to(self._vla_device)

        # Build action token IDs and append to input — matches original fine-tune
        action_token_ids = torch.stack([
            self._action_to_tokens(actions_gt[i].cpu().numpy())
            for i in range(B)
        ]).to(self._vla_device)

        full_input_ids = torch.cat([input_ids, action_token_ids], dim=1)
        full_attn_mask = torch.cat([
            attention_mask,
            torch.ones(B, self.ACTION_TOKEN_COUNT,
                       device=self._vla_device, dtype=attention_mask.dtype)
        ], dim=1)
        # Mask prompt positions with -100 so CE is only computed on action tokens
        labels = torch.cat([
            torch.full((B, prompt_len), -100,
                       dtype=torch.long, device=self._vla_device),
            action_token_ids,
        ], dim=1)

        # dummy_pv must match the ViT dtype (processed before the hook fires).
        # vit_tokens_dev must match projector.fc1.weight dtype (injected as backbone
        # output, fed directly into fc1).  A second hook casts projector output to
        # LLM dtype because the model has mixed fp16/bf16 precision across layers.
        _vit_dtype  = next(self._vla.vision_backbone.parameters()).dtype
        _fc1_dtype  = self._vla.projector.fc1.weight.dtype
        _llm_dtype  = next(self._vla.language_model.parameters()).dtype
        dummy_pv = torch.zeros(
            B, 6, 224, 224, device=self._vla_device, dtype=_vit_dtype
        )

        # Move vit_tokens to VLA device, keeping gradient
        vit_tokens_dev = vit_tokens.to(self._vla_device, dtype=_fc1_dtype)

        # Hook 1: replace backbone output with our tokens
        def _inject(module, inp, output):
            return vit_tokens_dev

        # Hook 2: cast projector output to LLM dtype
        def _cast_proj_out(module, inp, output):
            return output.to(dtype=_llm_dtype)

        h1 = self._vla.vision_backbone.register_forward_hook(_inject)
        h2 = self._vla.projector.register_forward_hook(_cast_proj_out)
        try:
            outputs = self._vla(
                input_ids      = full_input_ids,
                pixel_values   = dummy_pv,
                attention_mask = full_attn_mask,
                labels         = labels,
            )
        finally:
            h1.remove()
            h2.remove()

        return outputs.loss

    # ------------------------------------------------------------------ #
    #  predict_action_from_tokens  (inference)                            #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def predict_action_from_tokens(
        self,
        vit_tokens:   torch.Tensor,   # (B, N, D_vit)
        instructions: str | list,
    ) -> torch.Tensor:
        """
        Greedy-decode 7-DoF actions from reconstructed ViT tokens.
        Returns (B, 7) float32 on CPU.
        """
        self._load()
        B = vit_tokens.size(0)

        if isinstance(instructions, str):
            prompts = [self._make_prompt(instructions)] * B
        else:
            prompts = [self._make_prompt(inst) for inst in instructions]

        pil_dummy = [Image.new("RGB", (224, 224)) for _ in range(B)]
        inputs    = self._processor(
            prompts, pil_dummy, return_tensors="pt", padding=True
        )
        input_ids      = inputs["input_ids"].to(self._vla_device)
        attention_mask = inputs["attention_mask"].to(self._vla_device)
        _vit_dtype = next(self._vla.vision_backbone.parameters()).dtype
        _llm_dtype = next(self._vla.language_model.parameters()).dtype
        dummy_pv       = torch.zeros(
            B, 6, 224, 224, device=self._vla_device, dtype=_vit_dtype
        )
        vit_tokens_dev = vit_tokens.to(self._vla_device, dtype=_llm_dtype)

        current_slice = [None]

        def _inject(module, inp, output):
            return current_slice[0]

        handle = self._vla.vision_backbone.register_forward_hook(_inject)
        try:
            actions = []
            for i in range(B):
                current_slice[0] = vit_tokens_dev[i:i+1]
                single_input = {
                    "input_ids":      input_ids[i:i+1],
                    "attention_mask": attention_mask[i:i+1],
                }
                action_np = self._vla.predict_action(
                    **single_input,
                    pixel_values = dummy_pv[i:i+1],
                    unnorm_key   = self.unnorm_key,
                    do_sample    = False,
                )
                actions.append(torch.from_numpy(action_np).float())
        finally:
            handle.remove()

        return torch.stack(actions, dim=0)

    # ------------------------------------------------------------------ #
    #  encode_image_train  (LoRA ViT, with gradient)                      #
    # ------------------------------------------------------------------ #

    def encode_image_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Like encode_image() but WITHOUT torch.no_grad(), so gradients flow
        back through the LoRA ViT adapters.  Only call on rank 0 during
        Stage 1 training after add_lora() has been called.

        Parameters
        ----------
        x : (B, C, H, W) float32 [0,1]

        Returns
        -------
        tokens : (B, N, D_vit) float32, with grad_fn, on same device as x
        """
        self._load()
        pv     = self._preprocess_pixels(x)
        tokens = self._vla.vision_backbone(pv)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.dim() == 3 and tokens.size(1) == 257:
            tokens = tokens[:, 1:, :]
        return tokens.to(x.device, dtype=x.dtype)   # NOT detached

    @torch.no_grad()
    def encoder_drift(self, x: torch.Tensor) -> float:
        """
        Cosine similarity between original (LoRA disabled) and current LoRA
        ViT features on the same images.  Returns 1.0 if LoRA not loaded.
        Lower values = more drift from the pretrained ViT.
        """
        if not self._loaded or self._vla is None:
            return 1.0
        pv = self._preprocess_pixels(x)

        z_lora = self._vla.vision_backbone(pv)
        if isinstance(z_lora, (tuple, list)):
            z_lora = z_lora[0]
        if z_lora.dim() == 3 and z_lora.size(1) == 257:
            z_lora = z_lora[:, 1:, :]

        self._vla.vision_backbone.disable_adapter_layers()
        z_orig = self._vla.vision_backbone(pv)
        self._vla.vision_backbone.enable_adapter_layers()
        if isinstance(z_orig, (tuple, list)):
            z_orig = z_orig[0]
        if z_orig.dim() == 3 and z_orig.size(1) == 257:
            z_orig = z_orig[:, 1:, :]

        return F.cosine_similarity(
            z_lora.float().flatten(1), z_orig.float().flatten(1), dim=-1
        ).mean().item()

    # ------------------------------------------------------------------ #
    #  LoRA helpers                                                        #
    # ------------------------------------------------------------------ #

    def add_lora(
        self,
        r:                   int        = 16,
        r_vit:               int | None = None,
        alpha:               int        = 32,
        dropout:             float      = 0.05,
        llm_target_modules:  list[str]  = None,
        vit_target_modules:  list[str]  = None,
    ):
        """
        Apply LoRA adapters to the LLM and ViT inside the VLA.

        After this call only the LoRA parameters have requires_grad=True;
        all base-model weights remain frozen.

        Parameters
        ----------
        llm_target_modules : attention projection names in the LLM
                             (default: ["q_proj", "v_proj"])
        vit_target_modules : attention projection names in the ViT
                             (default: ["q_proj", "v_proj"])
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("Install peft:  pip install peft")

        self._load()

        if llm_target_modules is None:
            llm_target_modules = ["q_proj", "v_proj"]
        if vit_target_modules is None:
            vit_target_modules = ["q_proj", "v_proj"]

        def _apply_lora(submodule, target_modules, rank):
            cfg = LoraConfig(
                r              = rank,
                lora_alpha     = alpha,
                lora_dropout   = dropout,
                target_modules = target_modules,
                bias           = "none",
            )
            return get_peft_model(submodule, cfg)

        # Apply to LLM
        if hasattr(self._vla, "language_model"):
            self._vla.language_model = _apply_lora(
                self._vla.language_model, llm_target_modules, r
            )
        else:
            print("[OpenVLAAgent] WARNING: 'language_model' attribute not found; "
                  "LLM LoRA skipped.")

        # Apply to ViT backbone (skipped if vit_target_modules is None/empty)
        if vit_target_modules and hasattr(self._vla, "vision_backbone"):
            self._vla.vision_backbone = _apply_lora(
                self._vla.vision_backbone, vit_target_modules, r_vit if r_vit is not None else r
            )
        elif vit_target_modules:
            print("[OpenVLAAgent] WARNING: 'vision_backbone' attribute not found; "
                  "ViT LoRA skipped.")

        # Ensure only LoRA params are trainable
        for name, p in self._vla.named_parameters():
            if "lora_" not in name:
                p.requires_grad_(False)

        n_train = sum(p.numel() for p in self._vla.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self._vla.parameters())
        print(f"[OpenVLAAgent] LoRA applied.  Trainable: {n_train:,} / {n_total:,} params.")

        # Dispatch to multiple GPUs now that all weights are real CPU tensors.
        # (Deferred from _load() to avoid PEFT + device_map meta-device error.)
        if self._vla_device == "auto" and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                from accelerate import dispatch_model, infer_auto_device_map
                from accelerate.hooks import remove_hook_from_submodules
                # Remove any stale accelerate hooks before dispatching
                remove_hook_from_submodules(self._vla)
                max_memory = {i: "4000MiB" for i in range(n_gpus)}
                device_map = infer_auto_device_map(
                    self._vla,
                    max_memory = max_memory,
                    # Prevent splitting a single transformer block across devices
                    no_split_module_classes = [
                        "LlamaDecoderLayer",   # LLM blocks
                        "Block",               # DINOv2 ViT blocks (timm)
                        "SiglipEncoderLayer",  # SigLIP ViT blocks
                    ],
                )
                dispatch_model(self._vla, device_map=device_map)
                alloc = [torch.cuda.memory_allocated(i) / 1024**3 for i in range(n_gpus)]
                print(f"[OpenVLAAgent] Dispatched across {n_gpus} GPUs: "
                      + "  ".join(f"GPU{i}={alloc[i]:.1f}GiB" for i in range(n_gpus)))
            else:
                self._vla = self._vla.cuda()
            self._vla_device = "cuda:0"

    def set_trainable_lora(self, vit: bool = True, llm: bool = True):
        """Set which LoRA adapters have requires_grad=True."""
        if not self._loaded or self._vla is None:
            return
        for name, p in self._vla.named_parameters():
            if "lora_" not in name:
                continue
            if "vision_backbone" in name:
                p.requires_grad_(vit)
            elif "language_model" in name:
                p.requires_grad_(llm)
        n_train = sum(p.numel() for p in self._vla.parameters() if p.requires_grad)
        print(f"[OpenVLAAgent] set_trainable_lora(vit={vit}, llm={llm})  trainable={n_train:,}")

    def lora_parameters(self) -> list:
        """Return LoRA-only parameters (by name), excluding projector."""
        if not self._loaded or self._vla is None:
            return []
        return [p for n, p in self._vla.named_parameters() if "lora_" in n and p.requires_grad]

    def projector_parameters(self) -> list:
        """Return all projector parameters (for Phase 2 unfreezing)."""
        if not self._loaded or self._vla is None:
            return []
        return list(self._vla.projector.parameters())

    def set_projector_trainable(self, trainable: bool):
        for p in self._vla.projector.parameters():
            p.requires_grad_(trainable)
        n = sum(p.numel() for p in self._vla.projector.parameters())
        print(f"[OpenVLAAgent] projector {'unfrozen' if trainable else 'frozen'}  ({n:,} params)")

    def lora_state_dict(self) -> dict:
        """Return a CPU state dict containing all LoRA weights."""
        if not self._loaded or self._vla is None:
            return {}
        # Save by name pattern, NOT by requires_grad — requires_grad changes
        # between train/val passes and would produce an empty dict after validation.
        return {
            k: v.detach().cpu()
            for k, v in self._vla.named_parameters()
            if "lora_" in k
        }

    def load_lora_state_dict(self, state_dict: dict):
        """Load LoRA weights from a saved state dict."""
        if not self._loaded:
            self._load()
        current = dict(self._vla.named_parameters())
        loaded = 0
        for k, v in state_dict.items():
            if k in current:
                current[k].data.copy_(v.to(current[k].device))
                loaded += 1
        print(f"[OpenVLAAgent] LoRA state loaded ({loaded} tensors).")

    def set_instruction(self, s: str):
        self.instruction = s

    def set_unnorm_key(self, k: str):
        self.unnorm_key = k


# ========================================================================== #
#  CPU Stub (for tests and shape debugging)                                  #
# ========================================================================== #

class OpenVLAStub(nn.Module):
    """
    Lightweight CPU stub with the same interface as OpenVLAAgent.
    Returns dummy tokens from encode_image and differentiable MSE from
    task_loss_from_tokens.  No GPU or model download required.
    """

    def __init__(
        self,
        N_patches:    int = 256,
        D_vit:        int = 2176,
        action_dim:   int = 7,
        instruction:  str = "",
        unnorm_key:   str = "bridge_orig",
        **kwargs,
    ):
        super().__init__()
        self.N_patches   = N_patches
        self.D_vit       = D_vit
        self.action_dim  = action_dim
        self.instruction = instruction
        self.unnorm_key  = unnorm_key
        self._loaded     = True   # always "loaded" for the stub
        self._vla        = None

        # Tiny surrogate: mean-pool tokens → action prediction
        self._task_head = nn.Sequential(
            nn.Linear(D_vit, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        for p in self._task_head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Returns zero tokens of correct shape."""
        B = x.size(0)
        return torch.zeros(B, self.N_patches, self.D_vit, device=x.device, dtype=x.dtype)

    def encode_image_train(self, x: torch.Tensor) -> torch.Tensor:
        """Returns zero tokens with requires_grad (stub for LoRA ViT path)."""
        B = x.size(0)
        return torch.zeros(
            B, self.N_patches, self.D_vit, device=x.device, dtype=x.dtype,
            requires_grad=True,
        )

    def task_loss_from_tokens(
        self,
        vit_tokens:   torch.Tensor,
        instructions,
        actions_gt:   torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable surrogate; grad flows through vit_tokens."""
        return vit_tokens.float().mean()

    @torch.inference_mode()
    def predict_action_from_tokens(
        self,
        vit_tokens:   torch.Tensor,
        instructions,
    ) -> torch.Tensor:
        return self._task_head(vit_tokens.float().mean(dim=1)).cpu()

    # LoRA stubs (no-ops for testing)
    def add_lora(self, **kwargs):
        pass

    def set_trainable_lora(self, vit: bool = True, llm: bool = True):
        pass

    def lora_parameters(self) -> list:
        return []

    def lora_state_dict(self) -> dict:
        return {}

    def load_lora_state_dict(self, state_dict: dict):
        pass

    def set_instruction(self, s: str):
        self.instruction = s

    def set_unnorm_key(self, k: str):
        self.unnorm_key = k
