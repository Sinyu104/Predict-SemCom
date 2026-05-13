"""
ctrl_world_wrapper.py  —  CTRL-WORLD SVD world model for Predictive SemCom.

Follows the CTRL-WORLD paper (arxiv 2510.10125) exactly:

Training (EDM, CrtlWorld.forward())
-------------------------------------
  sigma ~ LogNormal(P_mean=0.7, P_std=1.6)
  c_skip = 1/(σ²+1),  c_out = -σ/(σ²+1)^0.5,  c_in = 1/(σ²+1)^0.5
  c_noise = log(σ)/4  — passed as UNet timestep embedding
  History  : noisy with small sigma_h ~ N(0, 0.3²)  (robustness augmentation)
  Future   : c_in-scaled noisy latents
  Cond lat : first future frame with small noise σ_cond ~ U(0, 0.2), 1/vae_scale scaled
  loss = E[||(c_out·pred + c_skip·noisy)_{future} − z_{future}||² · (σ²+1)/σ²]
  CFG dropout: 5% probability of zeroing action embedding

Inference (CtrlWorldDiffusionPipeline style)
----------------------------------------------
  Future frames: pure Gaussian noise → EulerDiscreteScheduler multi-step denoise
  History frames: clean, prepended AFTER scheduler.scale_model_input() on future
  Condition latent: last history frame (1/vae_scale) repeated across all frames
  scheduler.step() handles c_out/c_skip (v_prediction / EDM-compatible)

UNet input per frame: 8 channels = 4 noisy + 4 condition
Action cond: Action_encoder2 (B, T, action_dim) → (B, T, 1024) → frame-level cross-attn

Loading
-------
  stub mode (svd_path=None)  : random weights, for unit-testing without downloads
  full mode                  : load from stabilityai/stable-video-diffusion-img2vid
                               then optionally load CTRL-WORLD checkpoint for Action_encoder2
"""

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# ── CTRL-WORLD's modified UNet (adds frame_level_cond support) ─────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ctrl_world.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel


# ========================================================================== #
#  Action_encoder2  (exact replica of CTRL-WORLD's module, no text branch)   #
# ========================================================================== #

class ActionEncoder(nn.Module):
    """
    MLP that maps per-frame actions → 1024-dim conditioning tokens.

    (B, T, action_dim) → (B, T, 1024)

    Architecture: Linear → SiLU → Linear → SiLU → Linear  (Kaiming init)
    Matches CTRL-WORLD's Action_encoder2 without the optional CLIP text branch.
    """

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.net[2].weight, mode="fan_in", nonlinearity="relu")

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, T, action_dim) → (B, T, 1024)"""
        return self.net(action)


# ========================================================================== #
#  EDM helpers                                                                 #
# ========================================================================== #

def _edm_scalings(sigma: torch.Tensor):
    """Compute c_skip, c_out, c_in, c_noise from sigma (EDM formulation)."""
    c_skip  = 1.0 / (sigma ** 2 + 1.0)
    c_out   = -sigma / (sigma ** 2 + 1.0) ** 0.5
    c_in    = 1.0 / (sigma ** 2 + 1.0) ** 0.5
    c_noise = (sigma.log() / 4.0)
    return c_skip, c_out, c_in, c_noise


def _get_add_time_ids(
    fps: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    dtype: torch.dtype,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Produce the (B, 3) added-time-ids tensor expected by the SVD UNet."""
    ids = torch.tensor([[fps, motion_bucket_id, noise_aug_strength]], dtype=dtype, device=device)
    return ids.repeat(batch_size, 1)


# ========================================================================== #
#  CtrlWorldWrapper                                                            #
# ========================================================================== #

class CtrlWorldWrapper(nn.Module):
    """
    CTRL-WORLD SVD world model adapted for single-camera 224×224 Franka data.

    Parameters
    ----------
    action_dim    : int   Franka action dimension (7).
    num_history   : int   Number of history frames fed to the UNet (6 in CTRL-WORLD).
    num_pred      : int   Number of future frames to predict (1 for side-info use case).
    vae_scale     : float SD-VAE scaling factor (0.18215).
    fps           : int   FPS token for SVD time embedding.
    motion_bucket : int   Motion-bucket token for SVD time embedding.
    svd_path      : str | None  Path/HF hub ID for SVD pretrained weights.
                               None → stub mode (random weights, for testing).
    ctrl_ckpt     : str | None  Path to a CTRL-WORLD .pt checkpoint to load
                               Action_encoder2 weights from.
    freeze_unet   : bool  If True (default), freeze the SVD UNet weights.
    dtype         : torch.dtype  Mixed-precision dtype for UNet + Action_encoder2.

    Key interfaces
    --------------
    predict_next_latent(z_history, action, deterministic=True)
        1-step EDM forward; no torch.randn when deterministic=True.
        z_history : (B, num_history, 4, H, W)
        action    : (B, num_history+num_pred, action_dim)
        → z_pred  : (B, num_pred, 4, H, W)

    forward_edm(z_clip, a_clip)
        EDM training loss for fine-tuning.
        z_clip : (B, num_history+num_pred, 4, H, W)
        a_clip : (B, num_history+num_pred, action_dim)
        → scalar loss
    """

    def __init__(
        self,
        action_dim:          int   = 7,
        num_history:         int   = 6,
        num_pred:            int   = 1,
        vae_scale:           float = 0.18215,
        fps:                 int   = 7,
        motion_bucket:       int   = 127,
        svd_path:            str | None = None,
        ctrl_ckpt:           str | None = None,
        freeze_unet:         bool  = True,
        finetune_cross_attn: bool  = False,
        dtype:               torch.dtype = torch.float16,
    ):
        super().__init__()
        self.action_dim    = action_dim
        self.num_history   = num_history
        self.num_pred      = num_pred
        self.vae_scale     = vae_scale
        self.fps           = fps
        self.motion_bucket = motion_bucket
        self._dtype        = dtype

        # ── UNet ──────────────────────────────────────────────────────────
        if svd_path is not None:
            print(f"[CtrlWorldWrapper] Loading SVD UNet from {svd_path} …")
            from diffusers import StableVideoDiffusionPipeline
            pipe = StableVideoDiffusionPipeline.from_pretrained(svd_path, torch_dtype=dtype)
            # Swap in CTRL-WORLD's modified UNet (adds frame_level_cond support)
            unet = UNetSpatioTemporalConditionModel()
            missing, unexpected = unet.load_state_dict(pipe.unet.state_dict(), strict=False)
            if missing:
                print(f"  [CtrlWorldWrapper] UNet missing keys ({len(missing)}): {missing[:3]}")
            # Keep the SVD scheduler for inference (EulerDiscreteScheduler, EDM-compatible)
            self.scheduler = pipe.scheduler
            del pipe
            torch.cuda.empty_cache()
            print("[CtrlWorldWrapper] SVD UNet loaded.")
        else:
            print("[CtrlWorldWrapper] stub mode — random UNet weights.")
            unet = UNetSpatioTemporalConditionModel()
            from diffusers import EulerDiscreteScheduler
            self.scheduler = EulerDiscreteScheduler()

        self.unet = unet.to(dtype)

        if freeze_unet:
            self.unet.requires_grad_(False)
            # Gradient checkpointing recomputes activations during backward instead of
            # storing them. Cuts activation memory by ~50-70% at ~20% compute overhead.
            # Necessary because gradients still flow through the UNet to action_encoder.
            self.unet.enable_gradient_checkpointing()
            print("[CtrlWorldWrapper] UNet frozen + gradient checkpointing enabled.")

        # ── Action encoder (always trainable) ─────────────────────────────
        self.action_encoder = ActionEncoder(action_dim)

        # ── Optionally unfreeze UNet cross-attention layers ────────────────
        self.finetune_cross_attn = finetune_cross_attn
        if finetune_cross_attn:
            n = self._unfreeze_cross_attn()
            print(f"[CtrlWorldWrapper] UNet cross-attn (attn2) unfrozen: {n/1e6:.2f}M extra trainable params.")

        # ── Load CTRL-WORLD checkpoint (Action_encoder2 weights) ──────────
        if ctrl_ckpt is not None:
            self._load_ctrl_ckpt(ctrl_ckpt)

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    def _load_ctrl_ckpt(self, path: str):
        """Load Action_encoder2 weights from a CTRL-WORLD .pt checkpoint."""
        print(f"[CtrlWorldWrapper] Loading CTRL-WORLD checkpoint from {path} …")
        state = torch.load(path, map_location="cpu")
        # CTRL-WORLD checkpoints are full model state dicts; extract action_encoder
        ae_state = {}
        for k, v in state.items():
            if k.startswith("action_encoder.action_encode."):
                new_k = k.replace("action_encoder.action_encode.", "net.")
                ae_state[new_k] = v
            elif k.startswith("action_encoder.net."):
                new_k = k.replace("action_encoder.", "")
                ae_state[new_k] = v
        if ae_state:
            missing, _ = self.action_encoder.load_state_dict(ae_state, strict=False)
            print(f"  [CtrlWorldWrapper] Loaded action_encoder ({len(ae_state)} tensors, "
                  f"{len(missing)} missing).")
        else:
            print("  [CtrlWorldWrapper] No action_encoder keys found in checkpoint.")

    def save_action_encoder(self, path: str):
        torch.save(self.action_encoder.state_dict(), path)

    def load_action_encoder(self, path: str):
        self.action_encoder.load_state_dict(torch.load(path, map_location="cpu"))

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _unet_forward(
        self,
        input_latents: torch.Tensor,
        timestep,
        action_hidden: torch.Tensor,
        B:             int,
    ) -> torch.Tensor:
        """
        Run the SVD UNet with frame-level action conditioning.

        input_latents : (B, T_total, 8, H, W)   — 4 noisy + 4 condition per frame
        timestep      : (B,) float c_noise=log(sigma)/4 during training,
                        or scalar sigma from scheduler during inference
        action_hidden : (B, T_total, 1024)       — per-frame conditioning (CTRL-WORLD shape)
        """
        device = input_latents.device
        added_time_ids = _get_add_time_ids(
            self.fps, self.motion_bucket, 0.0,
            self._dtype, B, device,
        ).to(device)

        model_out = self.unet(
            input_latents.to(self._dtype),
            timestep,
            encoder_hidden_states=action_hidden,  # (B, T, 1024) — matches CTRL-WORLD
            added_time_ids=added_time_ids,
            frame_level_cond=True,
        ).sample                                  # (B, T, 4, H, W)
        return model_out.float()

    # ------------------------------------------------------------------ #
    #  DDIM inference (pure noise → multi-step denoise, CTRL-WORLD style)  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_next_latent(
        self,
        z_history: torch.Tensor,
        action:    torch.Tensor,
        n_steps:   int = 10,
    ) -> torch.Tensor:
        """
        Predict future latent frames from history + action.

        Matches CtrlWorldDiffusionPipeline from the CTRL-WORLD paper:
        - Future frames start from pure Gaussian noise
        - History frames prepended clean (NOT scaled by c_in) at each step
        - scheduler.scale_model_input applies c_in to future only
        - scheduler.step (EulerDiscreteScheduler) handles c_out/c_skip internally

        z_history : (B, num_history, 4, H, W)
        action    : (B, num_history+num_pred, action_dim)
        n_steps   : number of denoising steps (10 default, 25-50 for higher quality)
        → z_pred  : (B, num_pred, 4, H, W)
        """
        B, T_h, C, H, W = z_history.shape
        T_p   = self.num_pred
        T_tot = T_h + T_p
        device = z_history.device
        dtype  = self._dtype

        # Pure Gaussian noise for future frames (B, T_p, 4, H, W)
        x_future = torch.randn(B, T_p, C, H, W, device=device, dtype=dtype)

        # Condition latent: last history frame (clean, no noise at inference)
        # repeated across all T_tot frames, scaled by 1/vae_scale
        cond_frame  = (z_history[:, -1] / self.vae_scale).to(dtype)   # (B, 4, H, W)
        cond_latent = cond_frame.unsqueeze(1).expand(B, T_tot, C, H, W)  # (B, T_tot, 4, H, W)

        # Action conditioning — computed once, reused each denoising step
        action_hidden = self.action_encoder(action.float()).to(dtype)  # (B, T_tot, 1024)

        # Set up the SVD EulerDiscreteScheduler (same as CTRL-WORLD pipeline)
        self.scheduler.set_timesteps(n_steps, device=device)

        for t in self.scheduler.timesteps:
            # scale_model_input applies c_in = 1/sqrt(sigma²+1) to future latents
            future_scaled = self.scheduler.scale_model_input(x_future, t)  # (B, T_p, 4, H, W)

            # Prepend clean history (NOT c_in-scaled, matching CTRL-WORLD pipeline)
            all_latents = torch.cat([z_history.to(dtype), future_scaled], dim=1)  # (B, T_tot, 4, H, W)

            # Channel-concatenate condition latent (same as training)
            input_lats = torch.cat([all_latents, cond_latent], dim=2)  # (B, T_tot, 8, H, W)

            # UNet forward — pass scheduler timestep t directly (matches CTRL-WORLD pipeline)
            with torch.autocast(device_type="cuda", dtype=dtype):
                model_out = self._unet_forward(input_lats, t, action_hidden, B)

            # Discard history output, keep only future predictions
            noise_pred = model_out[:, T_h:].to(dtype)  # (B, T_p, 4, H, W)

            # Scheduler step applies c_out/c_skip to compute x0 then Euler update
            x_future = self.scheduler.step(noise_pred, t, x_future).prev_sample

        return x_future  # (B, T_p, 4, H, W)

    # ------------------------------------------------------------------ #
    #  DDPM x₀-prediction training objective  (CTRL-WORLD Eq. 3)         #
    # ------------------------------------------------------------------ #

    def forward_ddpm(
        self,
        z_clip: torch.Tensor,
        a_clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        EDM training objective matching CTRL-WORLD CrtlWorld.forward().

        Differences from naive DDPM:
        - Log-normal sigma: sigma = exp(N(P_mean=0.7, P_std=1.6))
        - History augmented with small noise (sigma_h ~ N(0, 0.3²)) for robustness
        - Condition latent = first FUTURE frame with small noise (sigma_cond ~ U(0, 0.2))
        - Weighted loss: loss_weight = (sigma²+1)/sigma²
        - x0 prediction: predict_x0 = c_out * model_pred + c_skip * noisy_latents
        - 5% classifier-free guidance dropout on action hidden

        z_clip : (B, num_history+num_pred, 4, H, W)
        a_clip : (B, num_history+num_pred, action_dim)
        → scalar weighted MSE loss on future frames
        """
        B, T_tot, C, H, W = z_clip.shape
        T_h    = self.num_history
        device = z_clip.device
        latents = z_clip.float()

        # ── EDM log-normal sigma (P_mean=0.7, P_std=1.6 from CTRL-WORLD) ──
        P_mean, P_std = 0.7, 1.6
        rnd_normal = torch.randn([B, 1, 1, 1, 1], device=device)
        sigma      = (rnd_normal * P_std + P_mean).exp()            # (B,1,1,1,1)
        c_skip     = 1 / (sigma ** 2 + 1)
        c_out      = -sigma / (sigma ** 2 + 1) ** 0.5
        c_in       = 1 / (sigma ** 2 + 1) ** 0.5
        c_noise    = (sigma.log() / 4).reshape([B])                  # (B,)
        loss_weight = (sigma ** 2 + 1) / sigma ** 2                  # (B,1,1,1,1)

        # Full noisy latents with future sigma (used in c_out/c_skip prediction)
        noisy_latents = latents + torch.randn_like(latents) * sigma  # (B, T_tot, 4, H, W)

        # ── History: small noise augmentation for robustness ──────────────
        sigma_h     = torch.randn([B, T_h, 1, 1, 1], device=device) * 0.3
        history     = latents[:, :T_h]
        noisy_history = 1 / (sigma_h ** 2 + 1) ** 0.5 * (history + sigma_h * torch.randn_like(history))

        # ── Future: c_in-scaled noisy latents ────────────────────────────
        noisy_future_cin = c_in * noisy_latents[:, T_h:]            # (B, T_p, 4, H, W)
        input_latents    = torch.cat([noisy_history, noisy_future_cin], dim=1)  # (B, T_tot, 4, H, W)

        # ── Condition latent: last HISTORY frame with small noise ─────────
        # Using the future frame leaks the target through the channel-concat path.
        # Must match predict_next_latent which uses z_history[:, -1] at inference.
        current_img  = latents[:, T_h - 1]                         # (B, 4, H, W)
        sigma_cond   = torch.rand([B, 1, 1, 1], device=device) * 0.2
        c_in_cond    = 1 / (sigma_cond ** 2 + 1) ** 0.5
        cond_noisy   = c_in_cond * (current_img + torch.randn_like(current_img) * sigma_cond)
        cond_latent  = (cond_noisy / self.vae_scale).unsqueeze(1).expand(B, T_tot, C, H, W)

        # Stack condition at channel dim: (B, T_tot, 8, H, W)
        input_latents = torch.cat([input_latents, cond_latent], dim=2)

        # ── Action conditioning with 5% CFG dropout ───────────────────────
        action_hidden = self.action_encoder(a_clip.float()).to(self._dtype)   # (B, T_tot, 1024)
        cfg_mask = (torch.rand(B, device=device) > 0.05).float().view(B, 1, 1)
        action_hidden = action_hidden * cfg_mask.to(self._dtype)

        # ── UNet forward — pass c_noise as timestep (EDM convention) ─────
        with torch.autocast(device_type="cuda", dtype=self._dtype):
            model_pred = self._unet_forward(
                input_latents.to(self._dtype), c_noise.to(self._dtype), action_hidden, B
            )
        model_pred = model_pred.float()

        # ── x0 prediction and weighted loss on future frames only ─────────
        predict_x0 = c_out * model_pred + c_skip * noisy_latents    # (B, T_tot, 4, H, W)
        z_future   = latents[:, T_h:]                               # (B, T_p, 4, H, W)
        loss = ((predict_x0[:, T_h:] - z_future) ** 2 * loss_weight).mean()
        return loss

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    def _unfreeze_cross_attn(self) -> int:
        """Unfreeze attn2 (cross-attention to encoder_hidden_states) in every transformer block.

        Weights are cast to fp32 so that GradScaler.unscale_() works correctly.
        Under the existing torch.autocast context the fp32 weights are temporarily
        cast to fp16 for matmuls, so forward-pass efficiency is unchanged.
        """
        n = 0
        for name, module in self.unet.named_modules():
            if name.endswith(".attn2"):
                module.float()           # fp32 storage → GradScaler-compatible gradients
                for p in module.parameters():
                    p.requires_grad_(True)
                    n += p.numel()
        return n

    def cross_attn_parameters(self):
        """Yield deduplicated parameters of UNet attn2 (cross-attention) layers."""
        seen = set()
        for name, module in self.unet.named_modules():
            if name.endswith(".attn2"):
                for p in module.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield p

    def unet_cross_attn_state_dict(self) -> dict:
        """Return a state dict containing only attn2 weights (for checkpointing)."""
        state = {}
        for name, module in self.unet.named_modules():
            if name.endswith(".attn2"):
                for pname, p in module.named_parameters():
                    state[f"{name}.{pname}"] = p.detach().cpu()
        return state

    def load_unet_cross_attn_state_dict(self, state: dict):
        """Load attn2 weights from a saved state dict."""
        for name, module in self.unet.named_modules():
            if name.endswith(".attn2"):
                for pname, p in module.named_parameters():
                    key = f"{name}.{pname}"
                    if key in state:
                        p.data.copy_(state[key].to(p.device, dtype=p.dtype))

    def trainable_parameters(self):
        """Yield action_encoder params and, if enabled, UNet cross-attn params."""
        yield from self.action_encoder.parameters()
        if self.finetune_cross_attn:
            yield from self.cross_attn_parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def to(self, *args, **kwargs):
        # Keep _dtype in sync when moving to a new dtype
        for a in args:
            if isinstance(a, torch.dtype):
                self._dtype = a
        return super().to(*args, **kwargs)


# ========================================================================== #
#  StubCtrlWorld — lightweight drop-in for CPU / stub testing                #
# ========================================================================== #

class StubCtrlWorld(nn.Module):
    """
    Trivial Ctrl-World replacement for CPU and stub-mode testing.

    Has the same interface as CtrlWorldWrapper but requires no GPU and no
    model downloads.  Predictions are simply the last history frame copied
    forward.  The DDPM loss is always zero (no parameters to update).

    Use with --use_stub or when no svd_path is provided.
    """

    def __init__(self, action_dim: int = 7, num_history: int = 6, num_pred: int = 1):
        super().__init__()
        self.action_dim  = action_dim
        self.num_history = num_history
        self.num_pred    = num_pred
        # Single dummy parameter so DDP wrapping doesn't complain
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    @torch.no_grad()
    def predict_next_latent(
        self,
        z_history: torch.Tensor,
        action:    torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return last history frame as prediction: (B, num_pred, C, H, W)."""
        last = z_history[:, -1:]                          # (B, 1, C, H, W)
        return last.expand(-1, self.num_pred, -1, -1, -1).clone()

    def forward_ddpm(
        self,
        z_clip: torch.Tensor,
        a_clip: torch.Tensor,
    ) -> torch.Tensor:
        """Zero loss (no trainable params)."""
        return z_clip.new_zeros(1).mean()

    def trainable_parameters(self):
        return iter([])

    def num_trainable_params(self) -> int:
        return 0

    def save_action_encoder(self, path: str):
        pass

    def load_action_encoder(self, path: str):
        pass
