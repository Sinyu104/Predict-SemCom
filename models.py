"""
models.py  —  Neural network modules for the Predictive Semantic Communication System.

Architecture
------------
Device (Robot):
    x_t → [ViT (agent, frozen)] → (B, N, D_vit)
        → [TokenEncoder] → z_t (B, latent_dim)
        → [JsccEncoder] → s_t ~ q(s_t|z_t), shape (B, D_jscc)
        → [RayleighChannel] → ŝ_t (B, D_jscc)

Edge Server:
    (z_{t-1}, a_{t-1}) → [Predictor] → ẑ_t^{pred} (B, latent_dim)
    ẑ_t^{pred} → [SideInfoEncoder] → p(ŝ_t|ẑ_t^{pred})   ← loss only, no data flow
    (ŝ_t, ẑ_t^{pred}) → [JsccDecoder] → ẑ_t (B, latent_dim)
    ẑ_t → [TokenDecoder] → (B, N, D_vit)
        → [OpenVLA Projector (agent, frozen)] → (B, N, D_model)
        → [LLM (agent, frozen)] → â_t

Rate Loss (Wyner-Ziv)
---------------------
KL( q(ŝ_t|z_t) || p(ŝ_t|ẑ_t^{pred}) )
  q = N(μ_enc, σ²_enc + σ²_n)    σ²_n explicit: physical channel noise
  p = N(μ_prior, σ²_prior)         σ²_n absorbed into learned σ²_prior

Modules
-------
1. TokenEncoder      (B,N,D_vit) → (B,latent_dim)
2. JsccEncoder       (B,latent_dim) → μ_enc, log_var_enc, s_t
3. SideInfoEncoder   (B,latent_dim) → μ_prior, log_var_prior   [loss only]
4. JsccDecoder       (B,D_jscc)+(B,latent_dim) → (B,latent_dim)
5. Predictor         (B,latent_dim)+(B,action_dim) → (B,latent_dim)
6. TokenDecoder      (B,latent_dim) → (B,N,D_vit)
7. RayleighChannel   (B,D_jscc) → (B,D_jscc)
8. SemComSystem      container + static rate_loss()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  1. TOKEN ENCODER                                                           #
# ========================================================================== #

class TokenEncoder(nn.Module):
    """
    Aggregate ViT patch tokens to compact latent: (B, N, D_vit) → (B, latent_dim)
    Mean-pools over the patch dimension, then applies a two-layer MLP.
    """

    def __init__(self, D_vit: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_vit, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, D_vit) → z: (B, latent_dim)"""
        return self.net(tokens.mean(dim=1))


# ========================================================================== #
#  2. JSCC ENCODER                                                            #
# ========================================================================== #

class JsccEncoder(nn.Module):
    """
    Probabilistic JSCC encoder: z_t → q(s_t|z_t) = N(μ_enc, σ²_enc)

    No access to ẑ_t^{pred} — Wyner-Ziv constraint.
    Reparameterisation trick during training; s_t = μ_enc at inference.
    """

    def __init__(self, latent_dim: int, D_jscc: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(hidden_dim, D_jscc)
        self.fc_log_var = nn.Linear(hidden_dim, D_jscc)

    def forward(
        self, z: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu_enc      : (B, D_jscc)
        log_var_enc : (B, D_jscc)
        s_t         : (B, D_jscc)  sampled (training) or mean (inference)
        """
        h       = self.shared(z)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10.0, 2.0)

        if sample and self.training:
            std = torch.exp(0.5 * log_var)
            s   = mu + std * torch.randn_like(std)
        else:
            s   = mu
        return mu, log_var, s


# ========================================================================== #
#  3. SIDE INFORMATION ENCODER                                                #
# ========================================================================== #

class SideInfoEncoder(nn.Module):
    """
    Conditional prior: ẑ_t^{pred} → p(ŝ_t|ẑ_t^{pred}) = N(μ_prior, σ²_prior)

    Used ONLY in the rate loss (KL term). No data flows through this module —
    it learns to predict the distribution of ŝ_t from ẑ_t^{pred} alone,
    enabling tight rate estimation via the variational Wyner-Ziv bound.
    """

    def __init__(self, latent_dim: int, D_jscc: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(hidden_dim, D_jscc)
        self.fc_log_var = nn.Linear(hidden_dim, D_jscc)

    def forward(
        self, z_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_pred: (B, latent_dim) — predictor output ẑ_t^{pred}

        Returns
        -------
        mu_prior      : (B, D_jscc)
        log_var_prior : (B, D_jscc)
        """
        h       = self.shared(z_pred)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10.0, 2.0)
        return mu, log_var


# ========================================================================== #
#  4. JSCC DECODER                                                            #
# ========================================================================== #

class JsccDecoder(nn.Module):
    """
    Wyner-Ziv decoder: (ŝ_t, ẑ_t^{pred}) → ẑ_t

    Conditions on both the noisy received signal and the predictor's side
    information, learning a non-linear recovery function.
    """

    def __init__(self, D_jscc: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_jscc + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, s_hat: torch.Tensor, z_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        s_hat  : (B, D_jscc)    received channel signal
        z_pred : (B, latent_dim) predictor side information ẑ_t^{pred}
        → ẑ_t  : (B, latent_dim)
        """
        return self.net(torch.cat([s_hat, z_pred], dim=-1))


# ========================================================================== #
#  5. PREDICTOR  (action-conditioned V-JEPA 2 style)                         #
# ========================================================================== #

class Predictor(nn.Module):
    """
    Action-conditioned world model — V-JEPA 2 style.

    Runs on the RECEIVER (edge server) only. Both context and target encoders
    (ViT + TokenEncoder) are frozen from Stage 1. Only the Predictor is
    trained in Stage 2 via L_world = ||ẑ_{t+1}^{pred} − sg(z_{t+1})||².

    Stateless MLP — no recurrent hidden state.
    (z_t, a_t) → ẑ_{t+1}^{pred}
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z : (B, latent_dim)   current latent z_t
        a : (B, action_dim)   action a_t
        → z_pred: (B, latent_dim)   predicted next latent ẑ_{t+1}^{pred}
        """
        return self.net(torch.cat([z, a], dim=-1))


# ========================================================================== #
#  6. TOKEN DECODER                                                           #
# ========================================================================== #

class TokenDecoder(nn.Module):
    """
    Reconstruct ViT patch tokens from compact latent: (B, latent_dim) → (B, N, D_vit)

    Inverse of TokenEncoder. Output feeds into the frozen OpenVLA Projector.
    """

    def __init__(self, latent_dim: int, N_patches: int, D_vit: int, hidden_dim: int):
        super().__init__()
        self.N_patches = N_patches
        self.D_vit     = D_vit
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N_patches * D_vit),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → tokens: (B, N_patches, D_vit)"""
        B = z.size(0)
        return self.net(z).view(B, self.N_patches, self.D_vit)


# ========================================================================== #
#  7. RAYLEIGH CHANNEL                                                        #
# ========================================================================== #

class RayleighChannel(nn.Module):
    """
    Flat Rayleigh fading channel: y = h·x + n,  h, n ~ CN(0,1)

    Signal power is estimated per-batch. Coherent equalization divides by |h|².
    """

    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db

    @property
    def sigma_n_sq(self) -> float:
        """Noise variance at unit signal power — used in the KL rate loss."""
        return 1.0 / (10.0 ** (self.snr_db / 10.0))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, D_jscc) → ŝ: (B, D_jscc)"""
        B = s.size(0)
        sig_power = s.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
        snr_lin   = 10.0 ** (self.snr_db / 10.0)
        noise_std = (sig_power / snr_lin).sqrt()

        h_re = torch.randn(B, 1, device=s.device) / math.sqrt(2)
        h_im = torch.randn(B, 1, device=s.device) / math.sqrt(2)

        y_re = h_re * s
        y_im = h_im * s
        n_re = noise_std * torch.randn_like(s) / math.sqrt(2)
        n_im = noise_std * torch.randn_like(s) / math.sqrt(2)

        h_mag_sq = (h_re.pow(2) + h_im.pow(2)).clamp(min=1e-8)
        return ((y_re + n_re) * h_re + (y_im + n_im) * h_im) / h_mag_sq


# ========================================================================== #
#  8. FULL SYSTEM (container)                                                 #
# ========================================================================== #

class SemComSystem(nn.Module):
    """
    Container for all trainable SemCom modules.

    ViT Encoder, OpenVLA Projector, and LLM are NOT stored here — they
    live in the OpenVLAAgent and are never passed to an optimiser.

    Each stage trainer uses a subset of these modules:
      Stage 1: token_encoder, token_decoder
      Stage 2: predictor  (token_encoder, token_decoder frozen)
      Stage 3: jscc_encoder, side_info_encoder, jscc_decoder, channel
               (all Stage 1/2 modules frozen)
    """

    def __init__(self, config: dict):
        super().__init__()
        D_vit      = config["D_vit"]
        latent_dim = config["latent_dim"]
        D_jscc     = config["D_jscc"]
        action_dim = config["action_dim"]
        hidden_dim = config["hidden_dim"]
        N_patches  = config["N_patches"]

        self.token_encoder     = TokenEncoder(D_vit, latent_dim, hidden_dim)
        self.jscc_encoder      = JsccEncoder(latent_dim, D_jscc, hidden_dim)
        self.side_info_encoder = SideInfoEncoder(latent_dim, D_jscc, hidden_dim)
        self.jscc_decoder      = JsccDecoder(D_jscc, latent_dim, hidden_dim)
        self.predictor         = Predictor(latent_dim, action_dim, hidden_dim)
        self.token_decoder     = TokenDecoder(latent_dim, N_patches, D_vit, hidden_dim)
        self.channel           = RayleighChannel(config["snr_db"])

    @staticmethod
    def rate_loss(
        mu_enc:        torch.Tensor,
        log_var_enc:   torch.Tensor,
        mu_prior:      torch.Tensor,
        log_var_prior: torch.Tensor,
        snr_db:        float,
    ) -> torch.Tensor:
        """
        KL( q(ŝ_t|z_t) || p(ŝ_t|ẑ_t^{pred}) )

        q: N(μ_enc,   σ²_enc + σ²_n)   — σ²_n = 1/snr_lin, explicit channel noise
        p: N(μ_prior, σ²_prior)         — σ²_n absorbed into learned σ²_prior

        Closed-form KL between two Gaussians (mean over batch and dim).
        """
        sigma_n_sq = 1.0 / (10.0 ** (snr_db / 10.0))
        q_var      = log_var_enc.exp() + sigma_n_sq          # (B, D_jscc)
        p_var      = log_var_prior.exp().clamp(min=1e-8)     # (B, D_jscc)

        kl = 0.5 * (
            p_var.log() - q_var.log()
            + (q_var + (mu_enc - mu_prior).pow(2)) / p_var
            - 1.0
        )
        return kl.mean()
