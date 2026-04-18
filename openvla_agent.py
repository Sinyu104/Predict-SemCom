"""
openvla_agent.py  —  OpenVLA wrapper for the Predictive SemCom system.

Two classes
-----------
OpenVLAAgent
    Wraps the real 7B model from HuggingFace.
    Supports two forward modes:
      • task_loss_forward()  — returns scalar MSE loss between predicted
                               and ground-truth actions.  Gradients flow
                               back through the reshaper and encoder.
                               Used in Stage 1 and Stage 2 training.
      • predict_action()     — returns (B, 7) action tensor.
                               Used at inference time.

    The VLA backbone weights are ALWAYS frozen.  Only the Encoder and
    Reshaper in the SemCom system receive gradients.

OpenVLAStub
    CPU-only placeholder.  Same interface as OpenVLAAgent.
    Used when:
      • Running unit tests without a GPU
      • Debugging shapes before the real model is downloaded

Why task_loss_forward() instead of a plain forward()
------------------------------------------------------
OpenVLA uses an autoregressive token-generation head.  The final action
is produced by argmax decoding, which is NOT differentiable.  To backprop
through the Reshaper, we use the language-model cross-entropy loss on the
ground-truth action tokens as a differentiable surrogate.  This is the
same approach used in the ATROC paper (Algorithm 2, line 9):

    ahat = OpenVLA(y_reconstructed)
    loss = task_loss(ahat, a_groundtruth)

We implement this as:
    logits = vla(input_ids, pixel_values=y_pixel)  → (B, T, vocab)
    loss   = cross_entropy(logits[:, action_pos, :], action_token_ids)

This loss propagates gradients through pixel_values → y → Reshaper → z.
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
    Frozen OpenVLA-7B wrapper with a differentiable task-loss path.

    Parameters
    ----------
    instruction : str    Language instruction, e.g. "pick up the red cube"
    unnorm_key  : str    Dataset key for action un-normalisation.
                         Use "franka_isaac" after fine-tuning, or
                         "bridge_orig" for zero-shot BridgeV2 domains.
    model_name  : str    HuggingFace model ID (default: openvla/openvla-7b)
    device      : str    Device for VLA weights (e.g. "cuda:0")
    quantize    : bool   4-bit loading via bitsandbytes (~6 GB VRAM)
    """

    ACTION_TOKEN_COUNT = 7   # OpenVLA predicts 7 action tokens

    def __init__(
        self,
        instruction: str  = "",
        unnorm_key:  str  = "franka_isaac",
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

        self._processor = None
        self._vla       = None
        self._loaded    = False

        # Dummy buffer so .to(device) works before lazy load
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
            # device_map="auto" distributes layers across all available GPUs.
            # .to() must NOT be called in this case — HuggingFace handles placement.
            kwargs["device_map"] = "auto"
            self._vla = AutoModelForVision2Seq.from_pretrained(
                self._model_name, **kwargs
            )
            # Resolve to a real device for subsequent tensor .to() calls.
            self._vla_device = next(self._vla.parameters()).device
        else:
            # Explicit device (e.g. "cuda:0") — load on CPU then move.
            self._vla = AutoModelForVision2Seq.from_pretrained(
                self._model_name, **kwargs
            ).to(self._vla_device)
        # Freeze ALL VLA parameters — gradients never flow into the VLA
        for p in self._vla.parameters():
            p.requires_grad_(False)
        self._vla.eval()
        self._loaded = True
        print("[OpenVLAAgent] Loaded and frozen.")

    # ------------------------------------------------------------------ #
    #  Tensor ↔ PIL helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_pil(t: torch.Tensor) -> Image.Image:
        """(3,H,W) float32 [0,1] tensor → PIL RGB image."""
        arr = (t.detach().cpu().clamp(0, 1) * 255).byte()
        return Image.fromarray(arr.permute(1, 2, 0).numpy(), "RGB")

    @staticmethod
    def _action_to_tokens(action_np: np.ndarray, processor) -> torch.Tensor:
        """
        Encode a ground-truth (7,) float32 action into the token IDs
        that OpenVLA would produce.  Used to compute cross-entropy loss.
        """
        # OpenVLA tokenises continuous actions into discrete bins (256 bins).
        # We replicate the normalisation used by the processor.
        action_str = processor.tokenizer.bos_token   # placeholder
        # Use the processor's built-in action tokenisation if available
        if hasattr(processor, "action_tokenizer"):
            token_ids = processor.action_tokenizer(action_np)
            return torch.tensor(token_ids, dtype=torch.long)
        # Fallback: uniform 256-bin quantisation over [-1, 1]
        bins      = torch.linspace(-1, 1, 256)
        action_t  = torch.from_numpy(action_np).float().clamp(-1, 1)
        token_ids = torch.bucketize(action_t, bins)
        return token_ids  # (7,)

    # ------------------------------------------------------------------ #
    #  Differentiable task loss (used in Stage 1 and Stage 2 training)   #
    # ------------------------------------------------------------------ #

    def task_loss_forward(
        self,
        y_rec:        torch.Tensor,        # (B, 3, H, W) float32, Reshaper output
        actions_gt:   torch.Tensor,        # (B, 7) float32, ground-truth actions
        instructions: list[str] | None = None,  # per-sample; falls back to self.instruction
    ) -> torch.Tensor:
        """
        Compute a DIFFERENTIABLE task loss.

        Gradients flow:
            cross_entropy_loss → logits → pixel_values (y_rec) → Reshaper → z

        The cross-entropy is computed on the ground-truth action tokens.
        This is the same surrogate used in ATROC Algorithm 2.

        Parameters
        ----------
        y_rec      : (B, 3, H, W)  Reshaper output in [0,1], float32
        actions_gt : (B, 7)        Ground-truth 7-DoF actions

        Returns
        -------
        loss : scalar tensor, differentiable w.r.t. y_rec
        """
        self._load()

        B = y_rec.size(0)
        if instructions is None:
            instructions = [self.instruction] * B
        prompts = [
            f"In: What action should the robot take to {instr}?\nOut:"
            for instr in instructions
        ]

        # ── 1. Differentiable pixel_values for the whole batch ──────────── #
        if y_rec.shape[-2:] != (224, 224):
            y_resized = F.interpolate(
                y_rec, size=(224, 224), mode="bilinear", align_corners=False
            )
        else:
            y_resized = y_rec                                # (B, 3, 224, 224)

        mean = torch.tensor([0.485, 0.456, 0.406],
                             device=y_rec.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                             device=y_rec.device).view(1, 3, 1, 1)
        pixel_values_grad = (y_resized - mean) / std        # (B, 3, 224, 224)
        # OpenVLA's prismatic backbone (DINOv2 + SigLIP) expects 6 channels.
        pixel_values_grad = torch.cat(
            [pixel_values_grad, pixel_values_grad], dim=1   # (B, 6, 224, 224)
        )
        pixel_values_grad = pixel_values_grad.to(
            self._vla_device, dtype=torch.float16
        )

        # ── 2. input_ids via batched processor call ──────────────────────── #
        # PIL images are only needed for the processor's tokenisation of the
        # text prompt; gradients do NOT flow through this path.
        pil_imgs = [self._to_pil(y_rec[i]) for i in range(B)]
        inputs   = self._processor(
            prompts, pil_imgs,
            return_tensors="pt", padding=True,
        )
        input_ids      = inputs["input_ids"].to(self._vla_device)       # (B, T)
        attention_mask = inputs["attention_mask"].to(self._vla_device)  # (B, T)

        # ── 3. Ground-truth action token IDs for the whole batch ─────────── #
        action_token_ids = torch.stack([
            self._action_to_tokens(actions_gt[i].cpu().numpy(), self._processor)
            for i in range(B)
        ]).to(self._vla_device)                                         # (B, 7)

        # ── 4. Single batched VLA forward pass ───────────────────────────── #
        outputs = self._vla(
            input_ids      = input_ids,
            pixel_values   = pixel_values_grad,
            attention_mask = attention_mask,
            output_hidden_states = False,
        )
        # outputs.logits: (B, T, vocab_size)
        action_logits = outputs.logits[:, -self.ACTION_TOKEN_COUNT:, :]  # (B, 7, V)
        action_logits    = action_logits.to(y_rec.device, dtype=y_rec.dtype)
        action_token_ids = action_token_ids.to(y_rec.device)

        # ── 5. Cross-entropy over all B×7 action tokens ──────────────────── #
        V = action_logits.size(-1)
        return F.cross_entropy(
            action_logits.reshape(-1, V),       # (B*7, V)
            action_token_ids.reshape(-1),        # (B*7,)
        )

    # ------------------------------------------------------------------ #
    #  Inference — returns numpy action (no gradient)                     #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def predict_action(
        self,
        y_rec:        torch.Tensor,         # (B, 3, H, W) float32
        instructions: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Greedy-decode 7-DoF actions from the Reshaper output.
        Returns (B, 7) float32 on CPU.
        """
        self._load()
        B = y_rec.size(0)
        if instructions is None:
            instructions = [self.instruction] * B
        actions = []
        for i in range(B):
            prompt  = (f"In: What action should the robot take to "
                       f"{instructions[i]}?\nOut:")
            pil_img = self._to_pil(y_rec[i])
            inputs  = self._processor(prompt, pil_img).to(
                self._vla_device, dtype=torch.float16
            )
            action_np = self._vla.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
            actions.append(torch.from_numpy(action_np).float())
        return torch.stack(actions, dim=0)      # (B, 7)

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

    task_loss_forward() returns MSE(predicted_action, actions_gt)
    using a tiny MLP, so gradients flow and shape checks pass without
    requiring GPU or model downloads.

    Set  use_stub=True  in PredictiveSemComSystem to use this.
    """

    def __init__(
        self,
        obs_channels: int = 3,
        obs_height:   int = 224,
        obs_width:    int = 224,
        action_dim:   int = 7,
        instruction:  str = "",
        unnorm_key:   str = "franka_isaac",
        **kwargs,                           # absorb extra OpenVLAAgent args
    ):
        super().__init__()
        self.action_dim  = action_dim
        self.instruction = instruction
        self.unnorm_key  = unnorm_key

        # Tiny CNN → MLP for a differentiable task-loss surrogate
        self._enc = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(obs_channels * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        # Freeze stub weights so they are never updated
        for p in self.parameters():
            p.requires_grad_(False)

    def task_loss_forward(
        self,
        y_rec:        torch.Tensor,
        actions_gt:   torch.Tensor,
        instructions: list[str] | None = None,  # accepted but unused by stub
    ) -> torch.Tensor:
        pred = self._enc(y_rec)
        return F.mse_loss(pred, actions_gt.to(y_rec.device))

    @torch.inference_mode()
    def predict_action(
        self,
        y_rec:        torch.Tensor,
        instructions: list[str] | None = None,  # accepted but unused by stub
    ) -> torch.Tensor:
        return self._enc(y_rec).cpu()

    def set_instruction(self, s: str):
        self.instruction = s

    def set_unnorm_key(self, k: str):
        self.unnorm_key = k
