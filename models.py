"""
models.py  —  Neural network modules for the Predictive Semantic Communication System.

Architecture (action-conditioned V-JEPA 2 + Wyner-Ziv JSCC)
-------------------------------------------------------------
Device (Robot):
    x_t → [ViT (agent, frozen)] → tokens_t (B, N, D_vit)
        → [JsccEncoder] → s_t ~ q(s_t|tokens_t), shape (B, D_jscc)
        → [RayleighChannel] → ŝ_t (B, D_jscc)

Edge Server:
    (tokens_{t-1}, a_{t-1}) → [Predictor] → ẑ_t^{pred} (B, N, D_vit)
    ẑ_t^{pred} → [SideInfoEncoder] → p(ŝ_t|ẑ_t^{pred})    ← loss only
    (ŝ_t, ẑ_t^{pred}) → [JsccDecoder] → token_hat (B, N, D_vit)
        → [OpenVLA Projector (agent, frozen)] → (B, N, D_model)
        → [LLM (agent, frozen)] → â_t

Predictor design — narrow transformer (V-JEPA 2)
-------------------------------------------------
The predictor works at full patch token resolution (B, N, D_vit).
It is intentionally narrow: tokens are projected to d_pred ≪ D_vit,
attention operates at d_pred (e.g. 384), then projected back.
The action is embedded and prepended as a conditioning token.

JsccDecoder design — cross-attention
-------------------------------------
The predicted tokens ẑ_t^{pred} serve as queries; the received signal
ŝ_t (projected to d_pred) is the single key/value token.  Each patch
token independently attends to the channel signal to compute a correction.

Rate Loss (Wyner-Ziv)
---------------------
KL( q(ŝ_t|tokens_t) || p(ŝ_t|ẑ_t^{pred}) )
  q = N(μ_enc,   σ²_enc + σ²_n)   σ²_n explicit
  p = N(μ_prior, σ²_prior)         σ²_n absorbed into learned σ²_prior

Modules
-------
1. JsccEncoder      (B,N,D_vit) → μ_enc, log_var_enc, s_t  [via mean pool + MLP]
2. SideInfoEncoder  (B,N,D_vit) → μ_prior, log_var_prior    [loss only, mean pool + MLP]
3. JsccDecoder      (B,D_jscc) + (B,N,D_vit) → (B,N,D_vit) [cross-attention]
4. Predictor        (B,N,D_vit) + (B,action_dim) → (B,N,D_vit) [narrow transformer]
5. RayleighChannel  (B,D_jscc) → (B,D_jscc)
6. SemComSystem     container + static rate_loss()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  1. JSCC ENCODER                                                            #
# ========================================================================== #

class JsccEncoder(nn.Module):
    """
    Probabilistic JSCC encoder: tokens_t → q(s_t|tokens_t) = N(μ_enc, σ²_enc)

    Mean-pools over the N patch dimension, then applies a two-layer MLP.
    No access to ẑ_t^{pred} — Wyner-Ziv constraint.
    """

    def __init__(self, D_vit: int, D_jscc: int, d_pred: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(D_vit, d_pred),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(d_pred, D_jscc)
        self.fc_log_var = nn.Linear(d_pred, D_jscc)

    def forward(
        self, tokens: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tokens : (B, N, D_vit)
        Returns μ_enc, log_var_enc, s_t — all (B, D_jscc)
        """
        h       = self.shared(tokens.mean(dim=1))       # (B, d_pred)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10.0, 2.0)

        if sample and self.training:
            std = torch.exp(0.5 * log_var)
            s   = mu + std * torch.randn_like(std)
        else:
            s   = mu
        return mu, log_var, s


# ========================================================================== #
#  2. SIDE INFORMATION ENCODER                                                #
# ========================================================================== #

class SideInfoEncoder(nn.Module):
    """
    Conditional prior: ẑ_t^{pred} → p(ŝ_t|ẑ_t^{pred}) = N(μ_prior, σ²_prior)

    Used ONLY in the rate loss (KL term).  No data flows through this module.
    Mean-pools predicted tokens before encoding.
    """

    def __init__(self, D_vit: int, D_jscc: int, d_pred: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(D_vit, d_pred),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(d_pred, D_jscc)
        self.fc_log_var = nn.Linear(d_pred, D_jscc)

    def forward(
        self, z_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_pred : (B, N, D_vit) — predictor output ẑ_t^{pred}
        Returns μ_prior, log_var_prior — both (B, D_jscc)
        """
        h       = self.shared(z_pred.mean(dim=1))       # (B, d_pred)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10.0, 2.0)
        return mu, log_var


# ========================================================================== #
#  3. JSCC DECODER  (cross-attention)                                         #
# ========================================================================== #

class JsccDecoder(nn.Module):
    """
    Wyner-Ziv decoder: (ŝ_t, ẑ_t^{pred}) → token_hat (B, N, D_vit)

    Each predicted patch token (query) attends to the received channel
    signal (key/value) to compute a per-token correction.  Residual
    connection adds the correction to the predictor's estimate.

    Architecture:
        q   = token_proj(ẑ_t^{pred})          (B, N, d_pred)
        kv  = signal_proj(ŝ_t).unsqueeze(1)   (B, 1, d_pred)
        q'  = q + CrossAttn(norm(q), kv, kv)
        q'' = q' + FFN(norm(q'))
        token_hat = ẑ_t^{pred} + out_proj(q'')
    """

    def __init__(self, D_jscc: int, D_vit: int, d_pred: int, n_heads: int = 8):
        super().__init__()
        self.token_proj  = nn.Linear(D_vit,  d_pred)
        self.signal_proj = nn.Linear(D_jscc, d_pred)
        self.norm1       = nn.LayerNorm(d_pred)
        self.cross_attn  = nn.MultiheadAttention(
            d_pred, n_heads, dropout=0.0, batch_first=True
        )
        self.norm2       = nn.LayerNorm(d_pred)
        self.ffn         = nn.Sequential(
            nn.Linear(d_pred, d_pred * 4),
            nn.GELU(),
            nn.Linear(d_pred * 4, d_pred),
        )
        self.out_proj    = nn.Linear(d_pred, D_vit)

    def forward(
        self, s_hat: torch.Tensor, z_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        s_hat  : (B, D_jscc)    received channel signal ŝ_t
        z_pred : (B, N, D_vit)  predictor side information ẑ_t^{pred}
        → token_hat : (B, N, D_vit)
        """
        # Project tokens and signal to working dim
        q  = self.token_proj(z_pred)                           # (B, N, d_pred)
        kv = self.signal_proj(s_hat).unsqueeze(1)              # (B, 1, d_pred)

        # Cross-attention: each token attends to the received signal
        delta, _ = self.cross_attn(self.norm1(q), kv, kv)
        q = q + delta                                          # residual

        # FFN
        q = q + self.ffn(self.norm2(q))

        # Project correction back to D_vit and add to predictor estimate
        return z_pred + self.out_proj(q)                       # (B, N, D_vit)


# ========================================================================== #
#  4. PREDICTOR  (narrow transformer — V-JEPA 2 style)                        #
# ========================================================================== #

class Predictor(nn.Module):
    """
    Action-conditioned narrow transformer predictor (V-JEPA 2 style).

    Operates at full patch token resolution (B, N, D_vit) but with a
    narrow working dimension d_pred ≪ D_vit (e.g. 384 vs 2176).
    The action is embedded and prepended as a conditioning token.

    Runs on the RECEIVER (edge server) only.  Both context and target
    encoders are the frozen ViT.  Only the Predictor is trained (Stage 2).

    Architecture:
        input_proj  : D_vit → d_pred
        action_embed: action_dim → d_pred  (prepended as 1 token)
        transformer : n_layers × TransformerEncoderLayer(d_pred)
        output_proj : d_pred → D_vit
        residual    : output = tokens + output_proj(transformer_out)
    """

    def __init__(
        self,
        D_vit:      int,
        action_dim: int,
        d_pred:     int = 384,
        n_layers:   int = 6,
        n_heads:    int = 8,
    ):
        super().__init__()
        self.input_proj   = nn.Linear(D_vit, d_pred)
        self.action_embed = nn.Linear(action_dim, d_pred)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_pred,
            nhead          = n_heads,
            dim_feedforward = d_pred * 4,
            dropout        = 0.0,
            batch_first    = True,
            norm_first     = True,   # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_pred, D_vit)

    def forward(self, tokens: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, N, D_vit)   context patch tokens (previous frame)
        a      : (B, action_dim) action taken
        → z_pred : (B, N, D_vit) predicted next-frame patch tokens
        """
        # Project tokens to narrow working dim
        x = self.input_proj(tokens)                            # (B, N, d_pred)

        # Embed action, prepend as extra conditioning token
        a_tok = self.action_embed(a).unsqueeze(1)              # (B, 1, d_pred)
        x     = torch.cat([a_tok, x], dim=1)                  # (B, N+1, d_pred)

        # Narrow transformer
        x = self.transformer(x)                                # (B, N+1, d_pred)

        # Drop action token, project back to D_vit
        x = x[:, 1:, :]                                        # (B, N, d_pred)
        return tokens + self.output_proj(x)                    # (B, N, D_vit)


# ========================================================================== #
#  5. RAYLEIGH CHANNEL                                                        #
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
        B         = s.size(0)
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
#  6. FULL SYSTEM (container)                                                 #
# ========================================================================== #

class SemComSystem(nn.Module):
    """
    Container for all trainable SemCom modules.

    ViT Encoder, OpenVLA Projector, and LLM are NOT stored here.

    Training stages:
      Stage 2: predictor  (ViT frozen)
      Stage 3: jscc_encoder, side_info_encoder, jscc_decoder, channel
               (predictor frozen)
    """

    def __init__(self, config: dict):
        super().__init__()
        D_vit      = config["D_vit"]
        D_jscc     = config["D_jscc"]
        d_pred     = config["d_pred"]
        action_dim = config["action_dim"]

        self.jscc_encoder      = JsccEncoder(D_vit, D_jscc, d_pred)
        self.side_info_encoder = SideInfoEncoder(D_vit, D_jscc, d_pred)
        self.jscc_decoder      = JsccDecoder(D_jscc, D_vit, d_pred)
        self.predictor         = Predictor(D_vit, action_dim, d_pred)
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
        KL( q(ŝ_t|tokens_t) || p(ŝ_t|ẑ_t^{pred}) )

        q: N(μ_enc,   σ²_enc + σ²_n)   σ²_n = 1/snr_lin, explicit
        p: N(μ_prior, σ²_prior)         σ²_n absorbed into learned σ²_prior
        """
        sigma_n_sq = 1.0 / (10.0 ** (snr_db / 10.0))
        q_var      = log_var_enc.exp() + sigma_n_sq
        p_var      = log_var_prior.exp().clamp(min=1e-8)
        kl = 0.5 * (
            p_var.log() - q_var.log()
            + (q_var + (mu_enc - mu_prior).pow(2)) / p_var
            - 1.0
        )
        return kl.mean()
