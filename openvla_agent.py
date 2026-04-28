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
        kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=True)
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

        # Inject custom norm_stats (e.g. "franka_isaac") saved during fine-tuning.
        # Mirrors the same logic in vla_server.py.
        import json as _json, os as _os
        stats_path = _os.path.join(self._model_name, "dataset_statistics.json")
        if _os.path.isfile(stats_path):
            with open(stats_path) as _f:
                extra_stats = _json.load(_f)
            if hasattr(self._vla, "norm_stats"):
                self._vla.norm_stats.update(extra_stats)
                print(f"[OpenVLAAgent] Loaded norm_stats from {stats_path}")

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

    @staticmethod
    def _action_to_tokens(action_np: np.ndarray, processor) -> torch.Tensor:
        """Encode ground-truth (7,) action into OpenVLA token IDs."""
        if hasattr(processor, "action_tokenizer"):
            return torch.tensor(
                processor.action_tokenizer(action_np), dtype=torch.long
            )
        bins     = torch.linspace(-1, 1, 256)
        action_t = torch.from_numpy(action_np).float().clamp(-1, 1)
        return torch.bucketize(action_t, bins)

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
        input_ids      = inputs["input_ids"].to(self._vla_device)
        attention_mask = inputs["attention_mask"].to(self._vla_device)

        # Dummy pixel values (will be discarded by the hook)
        dummy_pv = torch.zeros(
            B, 6, 224, 224, device=self._vla_device, dtype=torch.float16
        )

        # Move vit_tokens to VLA device, keeping gradient
        vit_tokens_dev = vit_tokens.to(self._vla_device, dtype=torch.float16)

        # Hook: replace backbone output with our tokens
        def _inject(module, inp, output):
            return vit_tokens_dev

        handle = self._vla.vision_backbone.register_forward_hook(_inject)
        try:
            outputs = self._vla(
                input_ids      = input_ids,
                pixel_values   = dummy_pv,
                attention_mask = attention_mask,
            )
        finally:
            handle.remove()

        # Cross-entropy on action tokens
        action_token_ids = torch.stack([
            self._action_to_tokens(actions_gt[i].cpu().numpy(), self._processor)
            for i in range(B)
        ]).to(vit_tokens.device)

        action_logits = outputs.logits[:, -self.ACTION_TOKEN_COUNT:, :]
        action_logits = action_logits.to(vit_tokens.device, dtype=vit_tokens.dtype)
        action_token_ids = action_token_ids.to(vit_tokens.device)

        V = action_logits.size(-1)
        return F.cross_entropy(
            action_logits.reshape(-1, V),
            action_token_ids.reshape(-1),
        )

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
        dummy_pv       = torch.zeros(
            B, 6, 224, 224, device=self._vla_device, dtype=torch.float16
        )
        vit_tokens_dev = vit_tokens.to(self._vla_device, dtype=torch.float16)

        actions = []
        for i in range(B):
            # Hook must inject only the i-th sample; the loop processes one at a time.
            def _inject(module, inp, output, _i=i):
                return vit_tokens_dev[_i:_i+1]

            handle = self._vla.vision_backbone.register_forward_hook(_inject)
            single_input = {
                "input_ids":      input_ids[i:i+1],
                "attention_mask": attention_mask[i:i+1],
            }
            try:
                action_np = self._vla.predict_action(
                    **single_input,
                    pixel_values = dummy_pv[i:i+1],
                    unnorm_key   = self.unnorm_key,
                    do_sample    = False,
                )
            finally:
                handle.remove()
            actions.append(torch.from_numpy(action_np).float())

        return torch.stack(actions, dim=0)

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

    def task_loss_from_tokens(
        self,
        vit_tokens:   torch.Tensor,
        instructions,
        actions_gt:   torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable MSE surrogate; grad flows through vit_tokens."""
        pred = self._task_head(vit_tokens.mean(dim=1))   # (B, action_dim)
        return F.mse_loss(pred, actions_gt.to(vit_tokens.device, dtype=vit_tokens.dtype))

    @torch.inference_mode()
    def predict_action_from_tokens(
        self,
        vit_tokens:   torch.Tensor,
        instructions,
    ) -> torch.Tensor:
        return self._task_head(vit_tokens.mean(dim=1)).cpu()

    def set_instruction(self, s: str):
        self.instruction = s

    def set_unnorm_key(self, k: str):
        self.unnorm_key = k
