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

Predictor design — narrow transformer (V-JEPA 2 style with 3D RoPE)
----------------------------------------------------------------------
Full patch resolution (B, N, D_vit); narrow working dim d_pred ≪ D_vit.
Transformer blocks use factored 3D RoPE (frame × height × width per head),
Pre-LN, and stochastic-depth drop-path.
The action is embedded and prepended as a conditioning token (position 0,0,0).

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
4. Predictor        (B,N,D_vit) + (B,action_dim) → (B,N,D_vit) [V-JEPA2 narrow transformer, 3D RoPE]
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
#  PREDICTOR HELPERS (V-JEPA 2 style)                                        #
# ========================================================================== #

def _drop_path(
    x: torch.Tensor, drop_prob: float, training: bool
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask  = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep_prob).floor_()
    return x.div(keep_prob) * mask


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


def _rope_rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Factored RoPE rotation matching V-JEPA 2's rotate_queries_or_keys.
    x   : (B, n_heads, N, D_chunk)
    pos : broadcastable to (B, n_heads, N)  — integer position indices
    """
    D     = x.shape[-1]
    omega = 1.0 / (10000.0 ** (torch.arange(D // 2, dtype=x.dtype, device=x.device) / (D / 2.0)))
    freq  = pos.unsqueeze(-1) * omega              # (..., N, D/2)
    cos   = freq.cos().repeat(1, 1, 1, 2)          # (..., N, D)
    sin   = freq.sin().repeat(1, 1, 1, 2)
    y     = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    rot   = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * cos + rot * sin


class _PredMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner    = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _PredAttention(nn.Module):
    """Self-attention with factored 3D RoPE (frame × height × width per head)."""

    def __init__(
        self,
        hidden_size: int,
        n_heads:     int,
        grid_size:   int,
        qkv_bias:    bool  = True,
        attn_drop:   float = 0.0,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = hidden_size // n_heads
        self.scale     = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.attn_drop = attn_drop

        self.q    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # Allocate equal head-dim budget to each spatial axis (rounded to even)
        self.d_dim = 2 * ((self.head_dim // 3) // 2)
        self.h_dim = 2 * ((self.head_dim // 3) // 2)
        self.w_dim = 2 * ((self.head_dim // 3) // 2)

    @torch.no_grad()
    def _pos_ids(self, N_total: int, device: torch.device):
        """(frame_ids, height_ids, width_ids) — shape (1, 1, N_total).
        Token 0 is the action token, assigned position (0, 0, 0).
        Tokens 1..N are patch tokens in raster scan order.
        """
        N_patch = N_total - 1
        tpf     = self.grid_size * self.grid_size
        idx     = torch.arange(N_patch, device=device)
        f_ids   = idx // tpf
        h_ids   = (idx % tpf) // self.grid_size
        w_ids   = idx % self.grid_size
        zero    = idx.new_zeros(1)
        def prepend(t):
            return torch.cat([zero, t]).view(1, 1, -1)
        return prepend(f_ids), prepend(h_ids), prepend(w_ids)

    def _apply_rope(self, qk: torch.Tensor, pos_ids: tuple) -> torch.Tensor:
        f_ids, h_ids, w_ids = pos_ids
        s1  = self.d_dim
        s2  = s1 + self.h_dim
        s3  = s2 + self.w_dim
        out = [
            _rope_rotate(qk[..., :s1], f_ids),
            _rope_rotate(qk[..., s1:s2], h_ids),
            _rope_rotate(qk[..., s2:s3], w_ids),
        ]
        if s3 < self.head_dim:
            out.append(qk[..., s3:])
        return torch.cat(out, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        def to_heads(lin):
            return lin(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = to_heads(self.q), to_heads(self.k), to_heads(self.v)

        pos = self._pos_ids(N, x.device)
        q   = self._apply_rope(q, pos)
        k   = self._apply_rope(k, pos)

        attn = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1,
                         dtype=torch.float32).to(x.dtype)
        if self.training and self.attn_drop > 0.0:
            attn = F.dropout(attn, p=self.attn_drop)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _PredBlock(nn.Module):
    """V-JEPA 2 transformer block: Pre-LN, 3D-RoPE self-attention, drop-path."""

    def __init__(
        self,
        hidden_size:    int,
        n_heads:        int,
        grid_size:      int,
        mlp_ratio:      float = 4.0,
        drop_path_rate: float = 0.0,
        qkv_bias:       bool  = True,
        attn_drop:      float = 0.0,
    ):
        super().__init__()
        self.norm1     = nn.LayerNorm(hidden_size)
        self.attn      = _PredAttention(hidden_size, n_heads, grid_size, qkv_bias, attn_drop)
        self.drop_path = _DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2     = nn.LayerNorm(hidden_size)
        self.mlp       = _PredMLP(hidden_size, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ========================================================================== #
#  4. PREDICTOR  (V-JEPA 2 — narrow transformer with factored 3D RoPE)       #
# ========================================================================== #

class Predictor(nn.Module):
    """
    Action-conditioned V-JEPA 2 predictor with factored 3D RoPE attention.

    Tokens are projected to d_pred, processed by narrow transformer blocks
    with per-head factored rotary embeddings (frame × height × width), then
    projected back to D_vit with a residual connection.

    The action is embedded as a conditioning token prepended to the patch
    sequence.  It receives RoPE position (0, 0, 0) and is discarded after
    the transformer.

    Architecture:
        input_proj  : D_vit → d_pred
        action_embed: action_dim → d_pred  (prepended; position 0,0,0)
        blocks      : n_layers × _PredBlock(d_pred, 3D-RoPE, drop-path)
        norm        : LayerNorm(d_pred)
        output_proj : d_pred → D_vit
        residual    : tokens + output_proj(norm_out[1:])

    Config keys added vs. original:
        grid_size  — spatial patch grid side length (default 14 for 224/16)
        grid_depth — temporal depth = frames_per_clip // tubelet_size (default 1)
    """

    def __init__(
        self,
        D_vit:          int,
        action_dim:     int,
        d_pred:         int   = 384,
        n_layers:       int   = 6,
        n_heads:        int   = 8,
        grid_size:      int   = 14,
        grid_depth:     int   = 1,
        mlp_ratio:      float = 4.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.input_proj   = nn.Linear(D_vit, d_pred)
        self.action_embed = nn.Linear(action_dim, d_pred)

        dp_rates = [
            drop_path_rate * i / max(n_layers - 1, 1)
            for i in range(n_layers)
        ]
        self.blocks = nn.ModuleList([
            _PredBlock(d_pred, n_heads, grid_size, mlp_ratio, dp_rates[i])
            for i in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(d_pred)
        self.output_proj = nn.Linear(d_pred, D_vit)

    def forward(self, tokens: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, N, D_vit)   context patch tokens (previous frame)
        a      : (B, action_dim) action taken
        → z_pred : (B, N, D_vit) predicted next-frame patch tokens
        """
        x     = self.input_proj(tokens)                        # (B, N, d_pred)
        a_tok = self.action_embed(a).unsqueeze(1)              # (B, 1, d_pred)
        x     = torch.cat([a_tok, x], dim=1)                  # (B, N+1, d_pred)

        for blk in self.blocks:
            x = blk(x)                                         # (B, N+1, d_pred)

        x = self.norm(x)[:, 1:]                                # (B, N, d_pred)  drop action token
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
        grid_size  = config.get("grid_size", 14)
        grid_depth = config.get("grid_depth", 1)

        self.jscc_encoder      = JsccEncoder(D_vit, D_jscc, d_pred)
        self.side_info_encoder = SideInfoEncoder(D_vit, D_jscc, d_pred)
        self.jscc_decoder      = JsccDecoder(D_jscc, D_vit, d_pred)
        self.predictor         = Predictor(D_vit, action_dim, d_pred,
                                           grid_size=grid_size, grid_depth=grid_depth)
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
