"""
vae_wrapper.py  —  Frozen SD-VAE encoder / decoder.

Uses stabilityai/sd-vae-ft-mse from diffusers (or any AutoencoderKL).
224×224 → 4×28×28 (8× spatial downsampling, 4 latent channels).

Latents are scaled by the VAE's config.scaling_factor (~0.18215) before
storage / transmission and un-scaled before decoding, matching SD convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEWrapper(nn.Module):
    """
    Thin wrapper around a pretrained AutoencoderKL (diffusers).

    All parameters are frozen.  encode() and decode() are the only public
    methods.  The encode path returns the *scaled* latent mean (deterministic
    encoding — no reparameterisation noise at inference).
    """

    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self._model_name = model_name
        self._vae = None
        self._scale = None
        self.register_buffer("_dummy", torch.zeros(1))

    # ------------------------------------------------------------------ #
    #  Lazy loader                                                         #
    # ------------------------------------------------------------------ #

    def _load(self):
        if self._vae is not None:
            return
        try:
            from diffusers import AutoencoderKL
        except ImportError:
            raise ImportError(
                "Install diffusers:  pip install diffusers>=0.26"
            )
        print(f"[VAEWrapper] Loading {self._model_name} …")
        self._vae = AutoencoderKL.from_pretrained(
            self._model_name, torch_dtype=torch.float32
        )
        for p in self._vae.parameters():
            p.requires_grad_(False)
        self._scale = self._vae.config.scaling_factor
        print(
            f"[VAEWrapper] Loaded  scale={self._scale:.5f}  "
            f"params={sum(p.numel() for p in self._vae.parameters()):,}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)  float32 [0, 1]
        Returns z : (B, C_vae, H_vae, W_vae)  scaled latent mean
        """
        self._load()
        dev = self._dummy.device
        vae = self._vae.to(dev)
        # SD-VAE expects [-1, 1]
        x_norm = x.to(dev) * 2.0 - 1.0
        dist = vae.encode(x_norm).latent_dist
        return (dist.mean * self._scale).to(x.device)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, C_vae, H_vae, W_vae)  scaled latent
        Returns x̃ : (B, 3, H, W)  float32 [0, 1]
        """
        self._load()
        dev = self._dummy.device
        vae = self._vae.to(dev)
        z_unscaled = z.to(dev) / self._scale
        out = vae.decode(z_unscaled).sample          # (B, 3, H, W), [-1, 1]
        return ((out.clamp(-1, 1) + 1.0) / 2.0).to(z.device)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if self._vae is not None:
            self._vae = self._vae.to(*args, **kwargs)
        return result
