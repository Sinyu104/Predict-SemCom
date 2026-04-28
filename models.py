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
#  TOKEN ENCODER / DECODER  (LeWorldModel-style compact latent)              #
# ========================================================================== #

class TokenEncoder(nn.Module):
    """
    Per-patch deterministic projection from frozen ViT space to compact space.

    Keeping the full (B, N, D_compact) spatial structure preserves fine-grained
    patch detail (important for small objects like the cube).
    No reparameterisation needed — SIGReg regularises the batch distribution.

    (B, N, D_vit) → c : (B, N, D_compact)
    """

    def __init__(self, D_vit: int, D_compact: int):
        super().__init__()
        self.fc = nn.Linear(D_vit, D_compact)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(tokens)


class TokenDecoder(nn.Module):
    """
    Per-patch projection from compact latent back to ViT token space.

    (B, N, D_compact) → (B, N, D_vit)
    """

    def __init__(self, D_compact: int, D_vit: int):
        super().__init__()
        self.fc = nn.Linear(D_compact, D_vit)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.fc(c)


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
    cos      = freq.cos().repeat(1, 1, 1, 2)           # (..., N, D) — half-split convention
    sin      = freq.sin().repeat(1, 1, 1, 2)
    x1, x2  = x.chunk(2, dim=-1)                      # first half, second half
    rot      = torch.cat((-x2, x1), dim=-1)
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
    """Self-attention with factored 3D RoPE.
    Position IDs are computed externally by Predictor and passed in,
    so this module is agnostic to clip length / sequence layout.
    Uses F.scaled_dot_product_attention for memory efficiency (Flash Attention
    when available, otherwise math kernel).
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads:     int,
        qkv_bias:    bool  = True,
        attn_drop:   float = 0.0,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = hidden_size // n_heads
        self.scale     = self.head_dim ** -0.5
        self.attn_drop = attn_drop

        self.q    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.d_dim = 2 * ((self.head_dim // 3) // 2)
        self.h_dim = 2 * ((self.head_dim // 3) // 2)
        self.w_dim = 2 * ((self.head_dim // 3) // 2)

    def _apply_rope(self, qk: torch.Tensor, pos_ids: tuple) -> torch.Tensor:
        f_ids, h_ids, w_ids = pos_ids
        s1 = self.d_dim
        s2 = s1 + self.h_dim
        s3 = s2 + self.w_dim
        out = [
            _rope_rotate(qk[..., :s1], f_ids),
            _rope_rotate(qk[..., s1:s2], h_ids),
            _rope_rotate(qk[..., s2:s3], w_ids),
        ]
        if s3 < self.head_dim:
            out.append(qk[..., s3:])
        return torch.cat(out, dim=-1)

    def forward(
        self,
        x:         torch.Tensor,
        pos_ids:   tuple,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x         : (B, N, C)
        pos_ids   : (f_ids, h_ids, w_ids), each (1, 1, N) — from Predictor
        attn_mask : (1, 1, N, N) bool, True=attend — None for full attention
        """
        B, N, C = x.shape
        def to_heads(lin):
            return lin(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = to_heads(self.q), to_heads(self.k), to_heads(self.v)

        q = self._apply_rope(q, pos_ids)
        k = self._apply_rope(k, pos_ids)

        dropout_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = attn_mask,
            dropout_p  = dropout_p,
            scale      = self.scale,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _PredBlock(nn.Module):
    """V-JEPA 2 transformer block: Pre-LN, 3D-RoPE self-attention, drop-path."""

    def __init__(
        self,
        hidden_size:    int,
        n_heads:        int,
        mlp_ratio:      float = 4.0,
        drop_path_rate: float = 0.0,
        qkv_bias:       bool  = True,
        attn_drop:      float = 0.0,
    ):
        super().__init__()
        self.norm1     = nn.LayerNorm(hidden_size)
        self.attn      = _PredAttention(hidden_size, n_heads, qkv_bias, attn_drop)
        self.drop_path = _DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2     = nn.LayerNorm(hidden_size)
        self.mlp       = _PredMLP(hidden_size, mlp_ratio)

    def forward(
        self,
        x:         torch.Tensor,
        pos_ids:   tuple,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), pos_ids, attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ========================================================================== #
#  4. PREDICTOR  (V-JEPA 2 — narrow transformer with factored 3D RoPE)       #
# ========================================================================== #

class Predictor(nn.Module):
    """
    V-JEPA 2-AC predictor — robotics paper spec (~300M, 24L, 16H, 1024d).

    Training uses multi-frame clips with block-causal attention (paper §3.1):
      - forward_clip()   : teacher-forcing over T frames (paper Eq. 2)
      - rollout_2step()  : 2-step autoregressive prediction (paper Eq. 3)

    Block-causal attention: frame k attends to frames 1..k only.
    RoPE: full 3D (frame × height × width) for patch tokens;
          temporal-only (h=w=0) for action/pose tokens — paper §3.1.

    Single-frame forward() is kept for Stage 2 (JSCC) where the predictor
    provides side information one step at a time.

    Architecture per frame block:
        [a_tok, (pose_tok,) patch_0 .. patch_{N-1}]
    Position IDs computed in Predictor and passed to each _PredBlock.
    """

    def __init__(
        self,
        D_vit:          int,
        action_dim:     int,
        d_pred:         int   = 1024,
        n_layers:       int   = 24,
        n_heads:        int   = 16,
        grid_size:      int   = 16,
        grid_depth:     int   = 1,
        mlp_ratio:      float = 4.0,
        drop_path_rate: float = 0.0,
        pose_dim:       int   = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.pose_dim   = pose_dim
        self.grid_size  = grid_size

        self.input_proj   = nn.Linear(D_vit,      d_pred)
        self.action_embed = nn.Linear(action_dim, d_pred)
        self.action_scale = nn.Parameter(torch.tensor(3.0))
        if pose_dim > 0:
            self.pose_embed = nn.Linear(pose_dim, d_pred)

        dp_rates = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            _PredBlock(d_pred, n_heads, mlp_ratio, dp_rates[i])
            for i in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(d_pred)
        self.output_proj = nn.Linear(d_pred, D_vit)

    # ------------------------------------------------------------------ #
    #  Position IDs and causal mask (computed once per forward call)      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _pos_ids(self, n_cond: int, N: int, T: int, device: torch.device):
        """
        Returns (f_ids, h_ids, w_ids), each (1, 1, T*n_per_frame).
        Conditioning tokens (action, pose): temporal position k, h=w=0.
        Patch tokens: full 3D position (frame k, row, col).
        """
        patch_idx = torch.arange(N, device=device)
        ph = patch_idx // self.grid_size
        pw = patch_idx % self.grid_size
        f_list, h_list, w_list = [], [], []
        for k in range(T):
            f_list += [torch.full((n_cond,), k, device=device), torch.full((N,), k, device=device)]
            h_list += [torch.zeros(n_cond, device=device, dtype=ph.dtype), ph]
            w_list += [torch.zeros(n_cond, device=device, dtype=pw.dtype), pw]
        f = torch.cat(f_list).view(1, 1, -1)
        h = torch.cat(h_list).view(1, 1, -1)
        w = torch.cat(w_list).view(1, 1, -1)
        return f, h, w

    @torch.no_grad()
    def _causal_mask(self, T: int, n_per_frame: int, device: torch.device):
        """
        Block-causal bool mask (1, 1, N_total, N_total).
        True = attend: frame k attends to all tokens in frames 0..k.
        """
        N_total  = T * n_per_frame
        frame_of = torch.arange(N_total, device=device) // n_per_frame
        mask = frame_of.unsqueeze(1) >= frame_of.unsqueeze(0)   # (N, N)
        return mask.unsqueeze(0).unsqueeze(0)                    # (1, 1, N, N)

    # ------------------------------------------------------------------ #
    #  Sequence builder                                                    #
    # ------------------------------------------------------------------ #

    def _build_seq(
        self,
        tokens_T:  torch.Tensor,
        actions_T: torch.Tensor,
        poses_T:   torch.Tensor | None,
    ):
        """
        tokens_T  : (B, T, N, D_vit)
        actions_T : (B, T, action_dim)  — a_T is zeros (dummy for last frame)
        poses_T   : (B, T, pose_dim) optional
        Returns: x (B, T*n_per_frame, d_pred), n_cond, n_per_frame
        """
        B, T, N, _ = tokens_T.shape
        x_p = self.input_proj(tokens_T.reshape(B * T, N, -1)).reshape(B, T, N, -1)
        a   = self.action_embed(actions_T)                           # (B, T, d_pred)

        # Broadcast action into every patch token so all 8 layers see the
        # action signal directly, rather than relying on attention to propagate
        # it from a single prepended token across 256 patches.
        x_p = x_p + a.unsqueeze(2) * self.action_scale              # (B, T, N, d_pred)

        if self.pose_dim > 0 and poses_T is not None:
            p      = self.pose_embed(poses_T)                        # (B, T, d_pred)
            blocks = torch.cat([a.unsqueeze(2), p.unsqueeze(2), x_p], dim=2)
            n_cond = 2
        else:
            blocks = torch.cat([a.unsqueeze(2), x_p], dim=2)
            n_cond = 1

        n_per_frame = n_cond + N
        return blocks.reshape(B, T * n_per_frame, -1), n_cond, n_per_frame

    # ------------------------------------------------------------------ #
    #  Forward passes                                                      #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        tokens: torch.Tensor,
        a:      torch.Tensor,
        pose:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Single-frame prediction — used in Stage 2 (JSCC side information).
        tokens : (B, N, D_vit)
        a      : (B, action_dim)
        → z_pred : (B, N, D_vit)
        """
        B, N, _ = tokens.shape
        x_p  = self.input_proj(tokens)
        cond = [self.action_embed(a).unsqueeze(1)]
        if self.pose_dim > 0 and pose is not None:
            cond.append(self.pose_embed(pose).unsqueeze(1))
        n_cond = len(cond)
        x = torch.cat(cond + [x_p], dim=1)                          # (B, n_cond+N, d_pred)

        pos_ids = self._pos_ids(n_cond, N, T=1, device=x.device)
        for blk in self.blocks:
            x = blk(x, pos_ids)                                      # no causal mask: T=1
        x = self.norm(x)[:, n_cond:]                                 # (B, N, d_pred)
        return self.output_proj(x)                                    # (B, N, D_vit) — γΔ

    def forward_clip(
        self,
        tokens:  torch.Tensor,
        actions: torch.Tensor,
        poses:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Multi-frame teacher-forcing forward (paper Eq. 2).

        tokens  : (B, T, N, D_vit)
        actions : (B, T-1, action_dim)  — a_1 .. a_{T-1}
        poses   : (B, T, pose_dim) optional

        Returns ẑ_{2..T} : (B, T-1, N, D_vit)
        """
        B, T, N, D_vit = tokens.shape

        a_pad     = tokens.new_zeros(B, 1, self.action_dim)
        actions_T = torch.cat([actions, a_pad], dim=1)               # (B, T, action_dim)

        x, n_cond, n_per_frame = self._build_seq(tokens, actions_T, poses)
        pos_ids    = self._pos_ids(n_cond, N, T, x.device)
        causal_msk = self._causal_mask(T, n_per_frame, x.device)

        for blk in self.blocks:
            x = blk(x, pos_ids, causal_msk)

        x = self.norm(x).reshape(B, T, n_per_frame, -1)
        patch_out = x[:, :, n_cond:, :]                              # (B, T, N, d_pred)
        preds     = self.output_proj(patch_out)                      # (B, T, N, D_vit) — γΔ
        return preds[:, :-1]                                         # (B, T-1, N, D_vit)

    def rollout_2step(
        self,
        tokens:  torch.Tensor,
        actions: torch.Tensor,
        poses:   torch.Tensor | None = None,
        gamma:   float = 1.0,
    ) -> torch.Tensor:
        """
        2-step autoregressive rollout.
        Model outputs γΔ; divide by gamma to recover Δ and reconstruct absolute tokens.

        tokens  : (B, T≥3, N, D_vit)
        actions : (B, 2, action_dim)
        Returns ẑ_3 : (B, N, D_vit)  — absolute token
        """
        # Step 1 — predict γΔ_2, reconstruct ẑ_2
        delta_2 = self.forward_clip(
            tokens[:, :2],
            actions[:, :1],
            poses[:, :2] if poses is not None else None,
        )[:, 0]                                                      # (B, N, D_vit)
        z_hat_2 = tokens[:, 0] + delta_2 / gamma

        # Step 2 — predict γΔ_3, reconstruct ẑ_3
        dummy = torch.zeros_like(tokens[:, :1])
        tokens_ar = torch.cat(
            [tokens[:, :1], z_hat_2.unsqueeze(1), dummy], dim=1
        )                                                            # (B, 3, N, D_vit)
        delta_3 = self.forward_clip(
            tokens_ar,
            actions[:, :2],
            poses[:, :3] if poses is not None else None,
        )[:, 1]                                                      # (B, N, D_vit)
        return z_hat_2 + delta_3 / gamma                             # (B, N, D_vit) — ẑ_3

    def rollout_3step(
        self,
        tokens:  torch.Tensor,
        actions: torch.Tensor,
        poses:   torch.Tensor | None = None,
        gamma:   float = 1.0,
    ) -> torch.Tensor:
        """
        3-step autoregressive rollout.

        tokens  : (B, T≥4, N, D_vit)
        actions : (B, 3, action_dim)    — a_1, a_2, a_3
        poses   : (B, 4, pose_dim) optional
        gamma   : scaling factor used during training (model outputs γΔ)

        Step 1: ẑ_2 = z_1 + pred/γ
        Step 2: ẑ_3 = ẑ_2 + pred/γ
        Step 3: predict γΔ_4  (raw model output, same space as L_tf)
        Returns (ẑ_3, γΔ_4) : absolute token before last step, and raw final prediction
        """
        dummy = torch.zeros_like(tokens[:, :1])

        # Step 1 — predict γΔ_2, reconstruct ẑ_2
        delta_2 = self.forward_clip(
            tokens[:, :2],
            actions[:, :1],
            poses[:, :2] if poses is not None else None,
        )[:, 0]                                                      # (B, N, D_vit)
        z_hat_2 = tokens[:, 0] + delta_2 / gamma

        # Step 2 — predict γΔ_3, reconstruct ẑ_3
        tokens_ar2 = torch.cat(
            [tokens[:, :1], z_hat_2.unsqueeze(1), dummy], dim=1
        )                                                            # (B, 3, N, D_vit)
        delta_3 = self.forward_clip(
            tokens_ar2,
            actions[:, :2],
            poses[:, :3] if poses is not None else None,
        )[:, 1]                                                      # (B, N, D_vit)
        z_hat_3 = z_hat_2 + delta_3 / gamma

        # Step 3 — predict γΔ_4, reconstruct ẑ_4
        tokens_ar3 = torch.cat(
            [tokens[:, :1], z_hat_2.unsqueeze(1),
             z_hat_3.unsqueeze(1), dummy], dim=1
        )                                                            # (B, 4, N, D_vit)
        delta_4 = self.forward_clip(
            tokens_ar3,
            actions[:, :3],
            poses[:, :4] if poses is not None else None,
        )[:, 2]                                                      # (B, N, D_vit)
        return z_hat_3, delta_4                                      # ẑ_3, γΔ_4


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
      Stage 1: token_encoder, predictor (in D_compact space), token_decoder
               LeWorldModel-style: L_pred + L_kl + L_recon
      Stage 2: jscc_encoder, side_info_encoder, jscc_decoder, channel
               (token_encoder, predictor, token_decoder frozen)
    """

    def __init__(self, config: dict):
        super().__init__()
        D_vit       = config["D_vit"]
        D_jscc      = config["D_jscc"]
        D_compact   = config.get("D_compact", 256)          # compact latent dim
        d_pred      = config["d_pred"]                      # predictor hidden dim
        jscc_d_pred = config.get("jscc_d_pred", 384)
        action_dim  = config["action_dim"]
        pose_dim    = config.get("pose_dim",       0)
        grid_size   = config.get("grid_size",     14)
        grid_depth  = config.get("grid_depth",     1)
        n_layers    = config.get("pred_n_layers", 24)
        n_heads     = config.get("pred_n_heads",  16)
        mlp_ratio   = config.get("pred_mlp_ratio", 4.0)
        drop_path   = config.get("pred_drop_path", 0.0)

        # Stage 1 modules — compact latent world model
        self.token_encoder = TokenEncoder(D_vit, D_compact)
        self.token_decoder = TokenDecoder(D_compact, D_vit)
        # Predictor now runs in D_compact space (input_proj: D_compact→d_pred,
        # output_proj: d_pred→D_compact)
        self.predictor     = Predictor(D_compact, action_dim, d_pred,
                                       n_layers=n_layers, n_heads=n_heads,
                                       grid_size=grid_size, grid_depth=grid_depth,
                                       mlp_ratio=mlp_ratio, drop_path_rate=drop_path,
                                       pose_dim=pose_dim)

        # Stage 2 modules — JSCC channel coding
        self.jscc_encoder      = JsccEncoder(D_vit, D_jscc, jscc_d_pred)
        self.side_info_encoder = SideInfoEncoder(D_vit, D_jscc, jscc_d_pred)
        self.jscc_decoder      = JsccDecoder(D_jscc, D_vit, jscc_d_pred)
        self.channel           = RayleighChannel(config["snr_db"])

    @staticmethod
    def sigreg(Z: torch.Tensor, M: int = 64) -> torch.Tensor:
        """
        Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544).

        Projects the batch of embeddings Z onto M random unit directions, then
        applies the Epps-Pulley normality test statistic along each 1-D projection.
        By the Cramér-Wold theorem, matching all 1-D marginals ≡ matching the
        full joint distribution.

        T(z) = √(2π)/n² · ΣΣ exp(-(zⱼ-zₖ)²/2)
               - 2√π/n · Σ exp(-zⱼ²/4)
               + √(2π/3)

        T = 0 iff z ~ N(0,1) (limiting result via char. function integration).

        Parameters
        ----------
        Z : (n, d)  batch of embeddings — typically mean-pooled compact tokens
        M : int     number of random sketch directions
        """
        import math
        n, d = Z.shape

        # M random unit vectors on S^{d-1}
        u = F.normalize(torch.randn(d, M, device=Z.device, dtype=Z.dtype), dim=0)

        # Project and standardise each direction to zero mean / unit std
        H = Z @ u                                          # (n, M)
        H = H - H.mean(dim=0, keepdim=True)
        H = H / (H.std(dim=0, keepdim=True) + 1e-8)

        # Epps-Pulley statistic (vectorised over M)
        sqrt_2pi   = math.sqrt(2.0 * math.pi)
        sqrt_pi    = math.sqrt(math.pi)
        sqrt_2pi_3 = math.sqrt(2.0 * math.pi / 3.0)

        # Pairwise squared distances: (n, n, M)
        D_sq      = (H.unsqueeze(1) - H.unsqueeze(0)).pow(2)
        pair_term = (sqrt_2pi / n ** 2) * torch.exp(-D_sq / 2).sum(dim=[0, 1])
        indiv_term = (2.0 * sqrt_pi / n) * torch.exp(-H.pow(2) / 4).sum(dim=0)

        T = pair_term - indiv_term + sqrt_2pi_3            # (M,)
        return T.mean()

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
