"""
models.py  —  Neural network modules for the VAE-latent Wyner-Ziv SemCom System.

Transmitter (Robot):
    z_t  = VAE_encode(x_t)                           ← frozen SD-VAE
    s_t  ~ q(s_t | z_t) = N(μ_enc, σ²_enc)           ← JsccEncoder
    s̃_t  = FadingChannel(s_t)                        ← awgn / rayleigh / cdl

Receiver (Edge Server):
    ẑ_t  = CtrlWorld([z̃_{t-km},...,z̃_{t-1}], a_t)   ← frozen Stage-1 (ctrl_world_wrapper.py)
    p(s̃_t | ẑ_t) = N(μ_prior, σ²_prior)              ← SideInfoEncoder  (KL loss only)
    z̃_t  = RefinementDiffusion(ẑ_t, s̃_t)             ← SDEdit start from ẑ_t; s̃_t cross-attn
    x̃_t  = VAE_decode(z̃_t)                            ← frozen SD-VAE  (vae_wrapper.py)
    â_t  = Policy(x̃_t)

Modules in this file
--------------------
1. JsccEncoder          z_t          → μ_enc, log_var_enc, s_t   (B, D_jscc)
2. SideInfoEncoder      ẑ_t          → μ_prior, log_var_prior     (B, D_jscc)
3. RefinementDiffusion  (ẑ_t, s̃_t)  → z̃_t   — second diffusion, s̃_t via cross-attn
4. FadingChannel        s_t          → s̃_t   — awgn / rayleigh / cdl (RayleighChannel alias)
5. SemComSystem         container for Stage-2 JSCC modules + static rate_loss()

Ctrl-World (first diffusion) lives in ctrl_world_wrapper.py.
VAE encoder/decoder live in vae_wrapper.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  Helpers                                                                    #
# ========================================================================== #

def _make_2d_sincos_pos(H: int, W: int, d_model: int, device, dtype) -> torch.Tensor:
    """2D sinusoidal positional embedding (H*W, d_model)."""
    assert d_model % 4 == 0
    d2 = d_model // 2
    omega = 1.0 / (10000.0 ** (torch.arange(d2 // 2, device=device, dtype=dtype) / (d2 // 2)))
    y_pos = torch.arange(H, device=device, dtype=dtype)
    x_pos = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_pos, x_pos, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    pe_y = torch.cat([torch.sin(yy[:, None] * omega), torch.cos(yy[:, None] * omega)], dim=-1)
    pe_x = torch.cat([torch.sin(xx[:, None] * omega), torch.cos(xx[:, None] * omega)], dim=-1)
    return torch.cat([pe_y, pe_x], dim=-1)                     # (H*W, d_model)


# ========================================================================== #
#  1. JSCC ENCODER                                                            #
# ========================================================================== #

class _SpatialCode(nn.Module):
    """Conv trunk producing a (c, g, g) GRID of code symbols instead of a flat vector.

    Why this exists.  The flatten->MLP encoder puts every one of the 784 spatial
    positions into every hidden unit, so no transmitted symbol corresponds to a
    location; the receiver then re-localises through global cross-attention over
    `n_sig` chunks carrying LEARNED (not geometric) positional embeddings.  That is
    the hardest possible way to send position, and it shows: measured live with a
    perfect predictor context, cube COLOUR survived (a global, low-frequency
    property a distributed code carries easily) while cube POSITION was off by
    12.9 px -- about one cube width -- and the target vanished outright on some
    frames.  The robot located the right cube and could not grasp it.

    Here symbol (k, i, j) is tied to image region (i, j), so moving a cube two
    cells left is a local change to a few symbols rather than a global recoding.

    Rate is UNCHANGED: c*g*g == D_jscc exactly, and the flattened output keeps the
    same (B, D_jscc) interface, so FadingChannel and the KL rate term are untouched.
    """

    def __init__(self, C_in: int, D_jscc: int, grid: int, width: int = 128):
        super().__init__()
        assert D_jscc % (grid * grid) == 0, \
            f"D_jscc={D_jscc} must be divisible by grid^2={grid*grid}"
        self.grid = grid
        self.c_out = D_jscc // (grid * grid)
        # STRIDE-1 throughout, pool exactly once at the end.  An earlier version
        # strided 28->14->7 and then pooled up to `grid`, which both upsampled for
        # grid>7 and averaged the cube away before the pool: a ~15 px cube is under
        # one cell wide at grid=8, so code(z) and code(z_hat) came out nearly equal
        # and the KL had nothing to bill.  Measured separation-to-variance ratio at
        # init: flat 0.106, grid=8 0.021 (5x worse), grid=16 0.309 (3x better).
        # Keep grid fine enough that a cell is no wider than the object that moves.
        self.trunk = nn.Sequential(
            nn.Conv2d(C_in, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(grid),
        )
        self.head_mu = nn.Conv2d(width, self.c_out, 1)
        self.head_lv = nn.Conv2d(width, self.c_out, 1)

    def forward(self, z):
        h = self.trunk(z)
        mu = self.head_mu(h).flatten(1)                       # (B, D_jscc)
        lv = self.head_lv(h).flatten(1).clamp(-10.0, 2.0)
        return mu, lv


class JsccEncoder(nn.Module):
    """
    Probabilistic JSCC encoder: z_t → q(s_t | z_t) = N(μ_enc, σ²_enc)

    Flattens the VAE latent → MLP → μ and log_var.
    No access to ẑ_t (Wyner-Ziv transmitter constraint).

    NOTE: this used to mean-pool over space (z.mean(dim=(-2,-1))), feeding only
    C_vae=4 numbers per frame into the MLP. Spatial mean-pooling is ~translation
    invariant, so it destroyed exactly the information that differs between frames:
    measured on the cube data, two random frames differ by 0.329 in the full latent
    but only 0.00046 after pooling — 0.14%. The encoder then had almost nothing to
    transmit, the rate stayed near zero and could not respond to disturbances.
    Flattening keeps the full spatial layout.
    """

    def __init__(self, C_vae: int, H_vae: int, W_vae: int, D_jscc: int, d_hidden: int = 512,
                 spatial: bool = False, grid: int = 8):
        super().__init__()
        self.spatial = spatial
        self.D_jscc = D_jscc
        if spatial:
            self.code = _SpatialCode(C_vae, D_jscc, grid)
            return
        self.in_dim = C_vae * H_vae * W_vae
        self.shared = nn.Sequential(
            nn.Linear(self.in_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(d_hidden, D_jscc)
        self.fc_log_var = nn.Linear(d_hidden, D_jscc)

    def forward(
        self, z: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z : (B, C_vae, H_vae, W_vae)
        Returns μ_enc, log_var_enc, s_t — all (B, D_jscc)
        """
        if self.spatial:
            mu, log_var = self.code(z)
        else:
            h       = z.flatten(1)                  # (B, C_vae*H_vae*W_vae)
            h       = self.shared(h)
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
    Conditional prior: ẑ_t → p(s̃_t | ẑ_t) = N(μ_prior, σ²_prior)

    Used ONLY in the rate loss (KL term). No data flows through this module
    at inference. Same architecture as JsccEncoder — it must match, since the two
    are compared by the KL; flattening one without the other would give the prior
    strictly less capacity than the posterior.
    """

    def __init__(self, C_vae: int, H_vae: int, W_vae: int, D_jscc: int, d_hidden: int = 512,
                 spatial: bool = False, grid: int = 8):
        super().__init__()
        self.spatial = spatial
        if spatial:
            # Must mirror JsccEncoder: the two are compared by the KL, so giving the
            # prior a different code layout than the posterior makes the rate term
            # measure the architecture mismatch rather than the disturbance.
            self.code = _SpatialCode(C_vae, D_jscc, grid)
            return
        self.in_dim = C_vae * H_vae * W_vae
        self.shared = nn.Sequential(
            nn.Linear(self.in_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(d_hidden, D_jscc)
        self.fc_log_var = nn.Linear(d_hidden, D_jscc)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_hat : (B, C_vae, H_vae, W_vae)
        Returns μ_prior, log_var_prior — both (B, D_jscc)
        """
        if self.spatial:
            return self.code(z_hat)
        h       = z_hat.flatten(1)                  # (B, C_vae*H_vae*W_vae)
        h       = self.shared(h)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10.0, 2.0)
        return mu, log_var


# ========================================================================== #
#  3. REFINEMENT DIFFUSION  (second diffusion — Wyner-Ziv JSCC decoder)      #
# ========================================================================== #

class _TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → 2-layer SiLU MLP → d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, d_model)"""
        half  = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        return self.mlp(emb)


class _RefineBlock(nn.Module):
    """Pre-LN self-attn + cross-attn on ẑ_t + SEPARATE cross-attn on s̃_t + FFN.

    s̃_t gets its own attention rather than being concatenated into the ẑ_t
    conditioning sequence. Concatenated, the single s̃_t token competed in one
    softmax against 784 spatially-aligned ẑ_t tokens and lost: measured channel
    contribution was 0.00%. A dedicated pathway means every block must consult
    it, with no winner-take-all against the prediction.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm_sig   = nn.LayerNorm(d_model)
        self.sig_attn   = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm3      = nn.LayerNorm(d_model)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                sig: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N,     d_model)  — patch tokens
        cond : (B, N,     d_model)  — ẑ_t spatial conditioning tokens
        sig  : (B, n_sig, d_model)  — received-signal tokens (own pathway)
        """
        xn = self.norm1(x)
        h, _ = self.self_attn(xn, xn, xn)
        x = x + h

        xn = self.norm2(x)
        h, _ = self.cross_attn(xn, cond, cond)
        x = x + h

        xn = self.norm_sig(x)
        h, _ = self.sig_attn(xn, sig, sig)
        x = x + h

        x = x + self.ffn(self.norm3(x))
        return x


class RefinementDiffusion(nn.Module):
    """
    Second diffusion process: (ẑ_t, s̃_t) → z̃_t  — true SDEdit formulation.

    Training: DDPM x₀-prediction on z_t (true data distribution).
        t'' ~ Uniform[1, noise_level]
        x_{t''} = sqrt(ᾱ_{t''}) z_t + sqrt(1-ᾱ_{t''}) ε    ← noise TARGET, not ẑ_t
        z̃_t     = denoiser(x_{t''}, t'', ẑ_t, s̃_t)          ← both ẑ_t and s̃_t condition
        L        = MSE(z̃_t, z_t)

    Inference: SDEdit — partially noise ẑ_t, DDIM denoise conditioned on ẑ_t and s̃_t.
        x_{t''} = sqrt(ᾱ_{t''}) ẑ_t + sqrt(1-ᾱ_{t''}) ε    ← start from noised ẑ_t
        DDIM denoise from t'' → 0, conditioned on (ẑ_t, s̃_t)

    The model learns p(z_t | ẑ_t, s̃_t). At inference, starting from noised ẑ_t
    is a good initialisation since ẑ_t ≈ z_t; the model corrects the residual
    guided by s̃_t. This is the correct SDEdit setup (Song et al. 2021).
    """

    def __init__(
        self,
        D_jscc:   int,
        C_vae:    int   = 4,
        H_vae:    int   = 28,
        W_vae:    int   = 28,
        d_model:  int   = 256,
        n_heads:  int   = 8,
        n_layers: int   = 4,
        sig_spatial: bool = False,
        sig_grid:    int  = 8,
        disagreement_gate: bool = False,
    ):
        super().__init__()
        self.C_vae = C_vae
        self.H_vae = H_vae
        self.W_vae = W_vae
        # Disagreement-gated fusion: instead of the fixed residual z_hat + delta (which
        # defaults to copying z_hat and needs an external deviation gate to know when to
        # correct), emit a per-location gate g in [0,1] and return z_hat + g * delta. g is
        # read from the token features, which have cross-attended to BOTH z_hat and s~, so
        # it can detect where the channel disagrees with the prediction. Calm frames:
        # channel agrees with z_hat -> g~0 -> copy (quiet, learned, no hyperparameter).
        # Disturbed: channel contradicts z_hat locally -> g~1 there -> trust the channel.
        # This makes adaptivity architectural rather than a tuned res_tau threshold.
        self.disagreement_gate = disagreement_gate

        # DDPM noise schedule (scaled-linear, matching SD/SVD)
        betas            = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        alphas_cumprod   = (1.0 - betas).cumprod(dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)  # (1000,)

        # Noisy-input patch projection
        self.patch_proj  = nn.Linear(C_vae, d_model)
        # ẑ_t patch projection (separate weights — different semantic role)
        self.zhat_proj   = nn.Linear(C_vae, d_model)
        # Received signal projection → single conditioning token
        # s̃_t as n_sig tokens rather than a single one, so its dedicated
        # cross-attention has something to select over. 256 -> 16 x 16.
        # sig_spatial: s̃ arrives as a (c, g, g) GRID whose cell (i,j) describes image
        # region (i,j), so the tokens get a GEOMETRIC 2D sin-cos embedding on that grid
        # -- the same construction as the latent tokens' pos_embed, just at lower
        # resolution. Cross-attention then does a local lookup instead of searching a
        # set of chunks whose positions it had to learn. Falls back to the original
        # 16 unanchored chunks when spatial coding is off.
        self.sig_spatial = sig_spatial
        if sig_spatial:
            self.sig_grid = sig_grid
            self.n_sig    = sig_grid * sig_grid
            self.sig_dim  = D_jscc // self.n_sig
            self.sig_proj = nn.Linear(self.sig_dim, d_model)
            self.register_buffer("sig_pos_buf",
                                 torch.zeros(1, self.n_sig, d_model), persistent=False)
            self._sig_pos_init = False
        else:
            self.n_sig       = 16
            self.sig_dim     = D_jscc // self.n_sig
            self.sig_proj    = nn.Linear(self.sig_dim, d_model)
            self.sig_pos     = nn.Parameter(torch.zeros(1, self.n_sig, d_model))
        self.t_embed     = _TimestepEmbedding(d_model)
        self.t_proj      = nn.Linear(d_model, d_model)

        # 2D positional embedding shared by noisy patches and ẑ_t patches (lazy init)
        self.register_buffer("pos_embed", torch.zeros(1, H_vae * W_vae, d_model), persistent=False)
        self._pos_init   = False

        # Transformer denoiser
        self.blocks   = nn.ModuleList([_RefineBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm     = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, C_vae)
        if disagreement_gate:
            # Per-location, per-channel gate head. Bias init negative so sigmoid starts
            # near ~0.1 (default toward copying z_hat); training opens it where the channel
            # earns a correction. Same features feed delta and gate, so the gate learns
            # from the same z_hat/s~ evidence the correction uses.
            self.gate_proj = nn.Linear(d_model, C_vae)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -2.0)   # sigmoid(-2) ~= 0.12

    def _init_pos(self, device, dtype):
        if not self._pos_init:
            pe = _make_2d_sincos_pos(self.H_vae, self.W_vae, self.pos_embed.shape[-1], device, dtype)
            self.pos_embed = pe.unsqueeze(0)
            self._pos_init = True

    def _denoise_x0(
        self,
        x_noisy: torch.Tensor,   # (B, C, H, W) — noisy input
        t:       torch.Tensor,   # (B,) long — timestep indices
        z_hat:   torch.Tensor,   # (B, C, H, W) — Ctrl-World prediction ẑ_t
        s_tilde: torch.Tensor,   # (B, D_jscc)  — received channel signal
        drop_zhat: torch.Tensor | None = None,   # (B,) bool — hide ẑ_t from the delta path
    ) -> torch.Tensor:
        """Predict clean x_0 given x_{t''}, t'', ẑ_t, and s̃_t.

        Conditioning sequence fed to cross-attention:
            [ẑ_t spatial tokens (N) | s̃_t token (1)]  →  (B, N+1, d_model)

        ẑ_t tokens provide spatial prior; s̃_t token carries channel correction.
        """
        B, C, H, W = x_noisy.shape
        N = H * W
        # ẑ_t-dropout: on the selected samples the CORRECTION must be computed from
        # s̃_t alone — ẑ_t is hidden from both the input path and the conditioning
        # tokens, but is KEPT as the residual base so the model still starts from the
        # prediction and never has to reconstruct the whole latent from the channel.
        # Without this the decoder converges to ignoring s̃_t: measured |dL/ds~| after
        # training was 500x smaller than at init, which starves the encoder of any
        # gradient telling it what to encode.
        if drop_zhat is not None:
            keep    = (~drop_zhat).to(x_noisy.dtype).view(B, 1, 1, 1)
            x_noisy = x_noisy * keep
            z_cond  = z_hat   * keep
        else:
            z_cond  = z_hat
        self._init_pos(x_noisy.device, x_noisy.dtype)
        pos = self.pos_embed.to(x_noisy.dtype)                            # (1, N, d)

        # Noisy input tokens
        patches = x_noisy.reshape(B, C, N).transpose(1, 2)               # (B, N, C)
        x = self.patch_proj(patches) + pos                                # (B, N, d)

        # Timestep conditioning: broadcast to each token
        t_emb = self.t_proj(self.t_embed(t)).unsqueeze(1)                 # (B, 1, d)
        x     = x + t_emb

        # Conditioning sequence: ẑ_t spatial tokens + s̃_t global token
        zhat_patches = z_cond.reshape(B, C, N).transpose(1, 2)           # (B, N, C)
        cond         = self.zhat_proj(zhat_patches) + pos                 # (B, N, d)
        if self.sig_spatial:
            # s̃ is a (c, g, g) grid: tokens must be ordered by LOCATION, so move the
            # channel axis last -- reshape(B, n_sig, c) would interleave channels and
            # positions and silently destroy the geometry this change exists to add.
            g = self.sig_grid
            sig_map = s_tilde.reshape(B, self.sig_dim, g, g)               # (B, c, g, g)
            sig_tok = sig_map.reshape(B, self.sig_dim, self.n_sig).transpose(1, 2)
            if not self._sig_pos_init:
                self.sig_pos_buf = _make_2d_sincos_pos(
                    g, g, self.sig_pos_buf.shape[-1], x_noisy.device, x_noisy.dtype
                ).unsqueeze(0)
                self._sig_pos_init = True
            sig = self.sig_proj(sig_tok) + self.sig_pos_buf.to(x_noisy.dtype)
        else:
            sig      = self.sig_proj(
                s_tilde.reshape(B, self.n_sig, self.sig_dim)) + self.sig_pos  # (B, n_sig, d)

        for blk in self.blocks:
            x = blk(x, cond, sig)
        x = self.norm(x)

        # RESIDUAL OUTPUT: predict the correction to ẑ_t, not the whole latent.
        # Emitting zeros now reproduces ẑ_t exactly. Previously the network had to
        # rebuild all C*H*W values from scratch just to say "change nothing", and the
        # leftover error was the measured -17.8% on calm frames. It also makes the
        # channel's job literally the Wyner-Ziv residual z_t - ẑ_t.
        delta = self.out_proj(x)                                           # (B, N, C)
        delta = delta.transpose(1, 2).reshape(B, C, H, W)
        if self.disagreement_gate:
            # Per-location gate in [0,1] from the same features: g*delta is applied only
            # where the network sees the channel disagree with the prediction. The gate is
            # stored for inspection (rate/adaptivity diagnostics) and does not affect grad.
            gate = torch.sigmoid(self.gate_proj(x)).transpose(1, 2).reshape(B, C, H, W)
            self.last_gate = gate.detach()
            return z_hat + gate * delta
        return z_hat + delta

    def forward_ddpm(
        self,
        z_t:         torch.Tensor,   # (B, C, H, W) — ground truth target
        z_hat:       torch.Tensor,   # (B, C, H, W) — Ctrl-World side info ẑ_t
        s_tilde:     torch.Tensor,   # (B, D_jscc)  — received signal s̃_t
        noise_level: int = 1000,
        per_sample:  bool = False,        # return (B,) per-sample loss, not a scalar
        zhat_dropout: float = 0.0,        # prob. of hiding ẑ_t from the delta path
    ) -> torch.Tensor:
        """
        Conditional-correction loss: noise ẑ_t (NOT z_t), condition on (ẑ_t, s̃_t),
        predict z_t. Training input distribution now matches `sdedit_refine` exactly.

        x_{t''} = sqrt(ᾱ_{t''}) ẑ_t + sqrt(1-ᾱ_{t''}) ε,   t'' ~ U[1, noise_level]
        L        = MSE( denoiser(x_{t''}, t'', ẑ_t, s̃_t),  z_t )

        Previously this noised z_t, following textbook SDEdit (train a denoiser on real
        data, start it from a noised guide at inference). That premise does not hold
        here: SDEdit assumes the guide is OFF the data manifold and denoising drags it
        on. ẑ_t comes from a diffusion model trained on these same latents, so it is
        already on-manifold — just wrong in a specific way. Noise-then-denoise therefore
        moved it sideways to a different plausible latent rather than toward z_t, and
        measured refinement was worse than ẑ_t itself at EVERY t'' from 10 to 250
        (-12% to -35%), with the channel contributing nothing (<0.3%) on disturbed and
        calm frames alike.

        Noising ẑ_t instead removes z_t from the input entirely, so the only route to
        the target is the conditioning — ẑ_t plus the received signal s̃_t. That makes
        this a conditional corrector, which is what the task actually is.
        """
        B  = z_t.shape[0]
        if noise_level <= 0:
            # Direct conditional correction, no noise injected at all. Measured best:
            # SDEdit noise is applied uniformly to all C*H*W elements, so it damages
            # the ~78% static background that ẑ_t already had right in order to fix a
            # small moving region. Sweeping t''=0,1,2,5,10 the error rose monotonically
            # with t'' in both disturbed and calm groups, so zero is the optimum.
            t0 = torch.zeros(B, dtype=torch.long, device=z_t.device)
            dz = (torch.rand(B, device=z_t.device) < zhat_dropout) if (
                 zhat_dropout > 0 and self.training) else None
            pred0 = self._denoise_x0(z_hat, t0, z_hat, s_tilde, drop_zhat=dz)
            if per_sample:
                return ((pred0 - z_t.detach()) ** 2).mean(dim=(1, 2, 3))
            return F.mse_loss(pred0, z_t.detach())
        t_max = min(int(noise_level), len(self.alphas_cumprod) - 1)
        t  = torch.randint(1, t_max + 1, (B,), device=z_t.device)
        ab = self.alphas_cumprod[t].view(B, 1, 1, 1).float()
        eps     = torch.randn_like(z_t)
        x_noisy = ab.sqrt() * z_hat + (1.0 - ab).sqrt() * eps            # noise ẑ_t (matches inference)
        dz = (torch.rand(B, device=z_t.device) < zhat_dropout) if (
             zhat_dropout > 0 and self.training) else None
        pred_x0 = self._denoise_x0(x_noisy, t, z_hat, s_tilde, drop_zhat=dz)
        if per_sample:
            return ((pred_x0 - z_t.detach()) ** 2).mean(dim=(1, 2, 3))
        return F.mse_loss(pred_x0, z_t.detach())

    @torch.no_grad()
    def sdedit_refine(
        self,
        z_hat:       torch.Tensor,   # (B, C, H, W) — ẑ_t from Ctrl-World
        s_tilde:     torch.Tensor,   # (B, D_jscc)  — s̃_t received signal
        noise_level: int = 250,      # SDEdit noise level t'' (must satisfy t'' < t')
        n_steps:     int = 10,       # DDIM denoising steps
    ) -> torch.Tensor:
        """
        SDEdit inference: partially noise ẑ_t at level t'', then DDIM denoise.

        The model learned p(z_t | ẑ_t, s̃_t) so ẑ_t conditions every step,
        while s̃_t steers the output toward the true z_t.
        At high SNR, s̃_t cleanly corrects toward z_t.
        At low SNR, output falls back toward ẑ_t (graceful degradation).
        """
        B       = z_hat.shape[0]
        if noise_level <= 0:                       # direct mode — single forward pass
            t0 = torch.zeros(B, dtype=torch.long, device=z_hat.device)
            return self._denoise_x0(z_hat, t0, z_hat, s_tilde)
        t_start = min(noise_level, len(self.alphas_cumprod) - 1)

        # SDEdit initialisation: noise ẑ_t (good prior, close to z_t)
        alpha_s = self.alphas_cumprod[t_start].float()
        eps     = torch.randn_like(z_hat)
        x       = alpha_s.sqrt() * z_hat + (1.0 - alpha_s).sqrt() * eps

        # DDIM timestep schedule
        step_size = max(1, t_start // n_steps)
        timesteps = list(range(t_start, 0, -step_size))

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, device=z_hat.device, dtype=torch.long)
            pred_x0 = self._denoise_x0(x, t_batch, z_hat, s_tilde)      # ẑ_t conditions every step

            t_next     = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_cur  = self.alphas_cumprod[t_cur].float()
            alpha_next = self.alphas_cumprod[t_next].float() if t_next > 0 \
                         else torch.tensor(1.0, device=x.device)

            # DDIM deterministic update
            eps_pred = (x - alpha_cur.sqrt() * pred_x0) / (1.0 - alpha_cur).sqrt().clamp(min=1e-8)
            x        = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * eps_pred

        return x   # ≈ pred_x0 at the last step

    @torch.no_grad()
    def masked_sample(
        self,
        z_hat:   torch.Tensor,   # (B,C,H,W) prediction prior
        s_tilde: torch.Tensor,   # (B,D_jscc) received signal
        mask:    torch.Tensor,   # (B,1,H,W) in [0,1]; 1=regenerate (cube), 0=keep z_hat
        noise_level: int = 500,
        n_steps: int = 20,
        guidance: float = 1.0,   # classifier-free guidance weight on s~ (1=plain cond, >1 amplifies s~)
        pure_noise_mask: bool = True,   # init the mask region from pure noise (generate fresh from s~)
    ) -> torch.Tensor:
        """RePaint-style masked diffusion sampling. Only the mask region is freely
        SAMPLED (sharp) from the denoiser; the kept region is locked to z_hat (re-noised
        each step so the denoiser sees a consistent latent). Sharp cube where the mask is
        open, background = z_hat untouched.

        guidance>1: classifier-free guidance on s~ — x0 = x0(∅) + g*(x0(s~)-x0(∅)) —
        amplifies the channel's contribution (needs s~-dropout training). pure_noise_mask:
        start the mask region from N(0,I) so the cube is generated from s~, not drifted
        from the wrong prior. Requires a DDPM x0-denoiser (twophase_masked.py)."""
        B = z_hat.shape[0]
        t_start = min(noise_level, len(self.alphas_cumprod) - 1)
        ab = self.alphas_cumprod.float()
        eps = torch.randn_like(z_hat)
        x_zhat = ab[t_start].sqrt() * z_hat + (1.0 - ab[t_start]).sqrt() * eps
        if pure_noise_mask:                                  # mask region: pure noise; background: noised z_hat
            x = mask * torch.randn_like(z_hat) + (1.0 - mask) * x_zhat
        else:
            x = x_zhat
        step = max(1, t_start // n_steps)
        timesteps = list(range(t_start, 0, -step))
        zero_s = torch.zeros_like(s_tilde)
        for i, t_cur in enumerate(timesteps):
            tb = torch.full((B,), t_cur, device=z_hat.device, dtype=torch.long)
            x0 = self._denoise_x0(x, tb, z_hat, s_tilde)
            if guidance != 1.0:                              # classifier-free guidance on s~
                x0_u = self._denoise_x0(x, tb, z_hat, zero_s)
                x0 = x0_u + guidance * (x0 - x0_u)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            a_c = ab[t_cur]
            a_n = ab[t_next] if t_next > 0 else torch.tensor(1.0, device=x.device)
            eps_pred = (x - a_c.sqrt() * x0) / (1.0 - a_c).sqrt().clamp(min=1e-8)
            x_model  = a_n.sqrt() * x0 + (1.0 - a_n).sqrt() * eps_pred            # freely sampled
            x_keep   = a_n.sqrt() * z_hat + (1.0 - a_n).sqrt() * torch.randn_like(z_hat)  # locked to z_hat
            x = mask * x_model + (1.0 - mask) * x_keep
        return mask * x + (1.0 - mask) * z_hat


# ========================================================================== #
#  3b. DISAGREEMENT-FUSION REFINER  (Wyner-Ziv residual, channel-only content) #
# ========================================================================== #

class _FusionBlock(nn.Module):
    """One disagreement-gated fusion step.

    z_hat is CONTEXT only (it forms the query and drives the gate); the channel
    tokens S are the sole source of CONTENT (they are the values of the channel
    read). h is a correction ACCUMULATOR seeded at zero, so z_hat contributes no
    latent values inside the block — it enters the output only as the residual
    base z_hat + r outside. This is the structural difference from
    RefinementDiffusion, whose delta draws values from z_hat too and is then
    post-gated (measured: channel ignored, 0.0005 effect).
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.q_proj     = nn.Linear(d_model, d_model)          # query built from the prior context P
        self.ln_h       = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.gate_mlp   = nn.Sequential(                       # disagreement -> per-location gate
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.ln_ff      = nn.LayerNorm(d_model)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)), nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.last_g = None

    def forward(self, h: torch.Tensor, P: torch.Tensor, S: torch.Tensor,
                sig_pad: torch.Tensor | None = None) -> torch.Tensor:
        # h: (B,N,d) correction accumulator | P: (B,N,d) prior context | S: (B,n_sig,d) channel tokens
        # sig_pad: (B,n_sig) bool, True = token NOT transmitted (ignored in the channel read) — for
        #          adaptive/variable-length codes where a calm frame sends fewer tokens.
        qP = self.q_proj(P)                                    # the "question", from the prediction only
        hn = self.ln_h(h)
        h  = h + self.self_attn(qP, hn, hn)[0]                # coordinate the correction spatially
        c  = self.cross_attn(qP, S, S, key_padding_mask=sig_pad)[0]  # channel READ — content = values of S only
        c  = torch.nan_to_num(c)                               # all-masked rows -> 0 (calm frame, nothing sent)
        g  = torch.sigmoid(self.gate_mlp(torch.cat([P, c, c - P], dim=-1)))  # (B,N,1) open where they disagree
        h  = h + g * c                                         # accumulate channel content, gated
        h  = h + self.ffn(self.ln_ff(h))
        self.last_g = g.detach()
        return h


class DisagreementFusionRefiner(nn.Module):
    """z̃ = z_hat + r, where r is a channel-sourced correction fused by
    disagreement gating. Drop-in for RefinementDiffusion: exposes _denoise_x0
    (ignoring the diffusion args x_noisy/t) and forward_ddpm.

    z_hat provides the base (the whole static scene, never re-synthesised) and the
    context (queries + gates); s̃ provides all correction content. out_proj is
    zero-initialised so training starts as the pure predictor (r=0 -> z̃=z_hat).
    """

    def __init__(self, D_jscc: int, C_vae: int = 4, H_vae: int = 28, W_vae: int = 28,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 4, sig_grid: int = 4):
        super().__init__()
        self.C_vae, self.H_vae, self.W_vae = C_vae, H_vae, W_vae
        self.N = H_vae * W_vae
        self.sig_grid = sig_grid
        self.n_sig    = sig_grid * sig_grid
        assert D_jscc % self.n_sig == 0, f"D_jscc {D_jscc} not divisible by n_sig {self.n_sig}"
        self.sig_dim  = D_jscc // self.n_sig

        self.zhat_ctx = nn.Linear(C_vae, d_model)              # prior -> context (queries + gate only)
        self.sig_proj = nn.Linear(self.sig_dim, d_model)       # channel tokens
        self.blocks   = nn.ModuleList([_FusionBlock(d_model, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, C_vae)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)   # start as identity

        self.register_buffer("pos2d",   torch.zeros(1, self.N, d_model),     persistent=False)
        self.register_buffer("sig_pos", torch.zeros(1, self.n_sig, d_model), persistent=False)
        self._pos_init = False
        self.last_gate = None

    def _init_pos(self, device, dtype):
        if not self._pos_init:
            self.pos2d   = _make_2d_sincos_pos(self.H_vae, self.W_vae,
                                               self.pos2d.shape[-1], device, dtype).unsqueeze(0)
            self.sig_pos = _make_2d_sincos_pos(self.sig_grid, self.sig_grid,
                                               self.sig_pos.shape[-1], device, dtype).unsqueeze(0)
            self._pos_init = True

    def forward(self, z_hat: torch.Tensor, s_tilde: torch.Tensor,
                sig_pad: torch.Tensor | None = None) -> torch.Tensor:
        # sig_pad: (B,n_sig) bool, True = token NOT transmitted (adaptive/variable-length codes).
        B, C, H, W = z_hat.shape
        N = H * W
        self._init_pos(z_hat.device, z_hat.dtype)
        P = self.zhat_ctx(z_hat.reshape(B, C, N).transpose(1, 2)) + self.pos2d.to(z_hat.dtype)   # (B,N,d)
        # channel tokens ordered by grid location (mirror RefinementDiffusion's sig layout)
        sig_map = s_tilde.reshape(B, self.sig_dim, self.sig_grid, self.sig_grid)
        sig_tok = sig_map.reshape(B, self.sig_dim, self.n_sig).transpose(1, 2)                   # (B,n_sig,sig_dim)
        S = self.sig_proj(sig_tok) + self.sig_pos.to(z_hat.dtype)
        h = torch.zeros(B, N, P.shape[-1], device=z_hat.device, dtype=z_hat.dtype)               # accumulator
        gates = []
        for blk in self.blocks:
            h = blk(h, P, S, sig_pad=sig_pad); gates.append(blk.last_g)
        r = self.out_proj(self.out_norm(h)).transpose(1, 2).reshape(B, C, H, W)                  # correction
        self.last_gate = torch.stack(gates, 0).mean(0).detach()                                  # (B,N,1)
        return z_hat + r                                                                          # residual base

    def _denoise_x0(self, x_noisy, t, z_hat, s_tilde, drop_zhat=None, sig_pad=None):
        """Diffusion-signature shim: x_noisy and t are ignored (single forward pass)."""
        if drop_zhat is not None:
            z_hat = z_hat * (~drop_zhat).to(z_hat.dtype).view(-1, 1, 1, 1)
        return self.forward(z_hat, s_tilde, sig_pad=sig_pad)

    def forward_ddpm(self, z_t, z_hat, s_tilde, noise_level: int = 0,
                     per_sample: bool = False, zhat_dropout: float = 0.0):
        if zhat_dropout > 0.0:
            m = (torch.rand(z_t.shape[0], device=z_t.device) < zhat_dropout).view(-1, 1, 1, 1)
            z_hat = z_hat * (~m)
        z_tilde = self.forward(z_hat, s_tilde)
        loss = ((z_tilde - z_t) ** 2).mean(dim=(1, 2, 3))
        return loss if per_sample else loss.mean()


# ========================================================================== #
#  4. FADING CHANNEL  (awgn / rayleigh / cdl)                                #
# ========================================================================== #

class FadingChannel(nn.Module):
    """
    Analog fading channel for deep-JSCC symbols with coherent zero-forcing
    (matched-filter) reception:

        y = h·s + n                         h = complex fading gain, n ~ CN(0, σ²_n)
        ŝ = Re(conj(h)·y) / |h|²  =  s + effective noise      (post-ZF, σ²_n/|h|²)

    `channel_type` selects how the per-symbol complex gain `h` is drawn — the
    only thing that changes between channels (the receiver is identical). This
    mirrors the Sionna SSCC baseline (`baselines/traditional/sionna_channel.py`),
    so the SNR axis is comparable across the two systems:

        "awgn"      h = 1                                    no fading (RD reference)
        "rayleigh"  h ~ CN(0,1), one gain per sample         flat / block fading
        "cdl"       h from a Sionna 3GPP TR-38.901 CDL model  frequency/time-selective

    All channels are normalized (E|h|² = 1), so N0 = sig_power / snr_lin for
    every channel type — matching the baseline convention.

    `cdl_kwargs` is forwarded to the Sionna CDL generator (model, delay_spread,
    carrier_frequency, min_speed, max_speed, ...); Sionna is imported lazily and
    is only required when `channel_type == "cdl"`.
    """

    CHANNELS = ("awgn", "rayleigh", "cdl")

    def __init__(
        self,
        snr_db: float = 10.0,
        channel_type: str = "rayleigh",
        cdl_kwargs: dict | None = None,
    ):
        super().__init__()
        channel_type = str(channel_type).lower()
        if channel_type not in self.CHANNELS:
            raise ValueError(
                f"channel_type must be one of {self.CHANNELS}, got '{channel_type}'."
            )
        self.snr_db       = snr_db
        self.channel_type = channel_type
        self.cdl_kwargs   = cdl_kwargs or {}
        self._cdl         = None      # lazily-built Sionna CDL gain generator

    @property
    def sigma_n_sq(self) -> float:
        return 1.0 / (10.0 ** (self.snr_db / 10.0))

    def _sample_gains(self, B: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h_re, h_im), each broadcastable to (B, D). E|h|² = 1."""
        if self.channel_type == "awgn":
            ones = torch.ones(B, 1, device=device)
            return ones, torch.zeros(B, 1, device=device)

        if self.channel_type == "rayleigh":
            # Flat / block fading: one complex gain per sample, shared across symbols.
            h_re = torch.randn(B, 1, device=device) / math.sqrt(2)
            h_im = torch.randn(B, 1, device=device) / math.sqrt(2)
            return h_re, h_im

        # cdl — per-symbol frequency-selective gains from Sionna TR-38.901.
        if self._cdl is None:
            from baselines.traditional.sionna_channel import _CDLGains
            self._cdl = _CDLGains(device=str(device), **self.cdl_kwargs)
        h = self._cdl(B * D).reshape(B, D).to(device)
        return h.real.contiguous(), h.imag.contiguous()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, D_jscc) → s̃: (B, D_jscc)

        Consecutive real values are paired into complex baseband symbols, so
        D_jscc reals occupy D_jscc/2 complex channel uses:

            x_k   = s_{2k} + j s_{2k+1}
            y_k   = h_k x_k + n_k ,   n_k ~ CN(0, N0),  N0 = E|x|^2 / snr_lin
            x̂_k   = y_k conj(h_k) / |h_k|^2            (coherent ZF)

        This makes the measured SNR equal the nominal `snr_db` exactly.  The
        earlier formulation put a REAL s on both quadratures with N0/2 of noise
        in each, then discarded the imaginary branch for AWGN (h_im = 0) — so
        only half the noise was ever seen and every measurement came out 3 dB
        optimistic.
        """
        B, D      = s.shape
        if D % 2 != 0:
            raise ValueError(
                f"D_jscc must be even to form complex symbols, got {D}.")
        sig_power = s.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
        snr_lin   = 10.0 ** (self.snr_db / 10.0)

        x_re, x_im = s[:, 0::2], s[:, 1::2]                    # (B, D/2) each
        # E|x|^2 = 2*sig_power, so N0 = 2*sig_power/snr_lin and each quadrature
        # carries N0/2 = sig_power/snr_lin.
        noise_std = (sig_power / snr_lin).sqrt()

        h_re, h_im = self._sample_gains(B, D // 2, s.device)

        # y = h x + n   (complex multiply)
        y_re = h_re * x_re - h_im * x_im + noise_std * torch.randn_like(x_re)
        y_im = h_re * x_im + h_im * x_re + noise_std * torch.randn_like(x_im)

        # ZF: x̂ = y conj(h) / |h|^2
        h_mag_sq = (h_re.pow(2) + h_im.pow(2)).clamp(min=1e-8)
        xh_re = ( y_re * h_re + y_im * h_im) / h_mag_sq
        xh_im = (-y_re * h_im + y_im * h_re) / h_mag_sq

        out = torch.empty_like(s)
        out[:, 0::2] = xh_re
        out[:, 1::2] = xh_im
        return out


class RayleighChannel(FadingChannel):
    """Backward-compatible flat Rayleigh fading channel (channel_type='rayleigh')."""

    def __init__(self, snr_db: float = 10.0):
        super().__init__(snr_db, channel_type="rayleigh")


# ========================================================================== #
#  5. FULL SYSTEM (Stage-2 JSCC container)                                   #
# ========================================================================== #

class SemComSystem(nn.Module):
    """
    Container for all Stage-2 trainable JSCC modules.

    Does NOT include Ctrl-World (first diffusion) — that lives in
    ctrl_world_wrapper.py and is loaded separately by Stage2Trainer.

    Stage 1: only Ctrl-World action encoder is trained (not this system).
    Stage 2: jscc_encoder, side_info_encoder, refinement_diffusion are trained;
             channel is used but has no parameters.
    """

    def __init__(self, config: dict):
        super().__init__()
        C_vae    = config.get("C_vae",    4)
        H_vae    = config.get("H_vae",   28)
        W_vae    = config.get("W_vae",   28)
        D_jscc   = config["D_jscc"]
        d_hidden = config.get("jscc_d_hidden", 512)
        snr_db   = config["snr_db"]
        refine_d = config.get("refine_d_model",  256)
        refine_h = config.get("refine_n_heads",    8)
        refine_l = config.get("refine_n_layers",   4)

        spatial = config.get("jscc_spatial", False)
        grid    = config.get("jscc_grid", 8)

        self.jscc_encoder         = JsccEncoder(C_vae, H_vae, W_vae, D_jscc, d_hidden,
                                                spatial=spatial, grid=grid)
        self.side_info_encoder    = SideInfoEncoder(C_vae, H_vae, W_vae, D_jscc, d_hidden,
                                                    spatial=spatial, grid=grid)
        if config.get("refine_arch", "residual") == "fusion":
            self.refinement_diffusion = DisagreementFusionRefiner(
                D_jscc, C_vae, H_vae, W_vae, refine_d, refine_h, refine_l,
                sig_grid=config.get("refine_sig_grid", 4),
            )
        else:
            self.refinement_diffusion = RefinementDiffusion(
                D_jscc, C_vae, H_vae, W_vae, refine_d, refine_h, refine_l,
                sig_spatial=spatial, sig_grid=grid,
                disagreement_gate=config.get("refine_disagreement_gate", False),
            )

        channel_type = config.get("channel_type", "rayleigh")
        cdl_kwargs = {
            "model":             config.get("cdl_model",        "C"),
            "delay_spread":      config.get("cdl_delay_spread", 100e-9),
            "carrier_frequency": config.get("cdl_carrier_freq", 3.5e9),
            "max_speed":         config.get("cdl_max_speed",    3.0),
        }
        self.channel = FadingChannel(snr_db, channel_type=channel_type,
                                     cdl_kwargs=cdl_kwargs)

    @staticmethod
    def rate_loss(
        mu_enc:        torch.Tensor,
        log_var_enc:   torch.Tensor,
        mu_prior:      torch.Tensor,
        log_var_prior: torch.Tensor,
        snr_db:        float,
        per_sample:    bool = False,      # return (B,) per-sample KL, not a scalar
    ) -> torch.Tensor:
        """
        KL( q(s̃_t | z_t) || p(s̃_t | ẑ_t) )    — over the received signal s̃_t

        q : N(μ_enc,   σ²_enc + σ²_n)   where σ²_n = 1/snr_lin (physical channel noise)
        p : N(μ_prior, σ²_prior)         σ²_n absorbed into learned σ²_prior
        """
        sigma_n_sq = 1.0 / (10.0 ** (snr_db / 10.0))
        q_var      = log_var_enc.exp() + sigma_n_sq
        p_var      = log_var_prior.exp().clamp(min=1e-8)
        kl = 0.5 * (
            p_var.log() - q_var.log()
            + (q_var + (mu_enc - mu_prior).pow(2)) / p_var
            - 1.0
        )
        return kl.mean(dim=-1) if per_sample else kl.mean()


# ========================================================================== #
#  6. ADAPTIVE-RATE (MoE-ROUTER) WYNER-ZIV JSCC                              #
#                                                                            #
#  Transmitter routes each frame to a code LENGTH from the temporal          #
#  difference ||z_t - z_{t-1}|| (Wyner-Ziv clean: no prediction at the TX).  #
#  Validated separability: disturbed frames jump ~4x ordinary motion, p10    #
#  (disturbed) > p99 (ordinary); a threshold ~0.06 catches 96% at 1% FP.     #
#  Calm -> few tokens (channel quiet -> gate closed -> z̃=z_hat); disturbed   #
#  -> many tokens (channel carries the cube -> gate opens). The routing      #
#  level is the CLEAN disturbance signal the receiver keys on, replacing the #
#  global beta trade-off. Nested/prefix tokens: level L sends the first      #
#  n_L tokens, so one encoder serves all lengths.                            #
# ========================================================================== #

def _prefix_mask(level_probs: torch.Tensor, level_tokens, max_tokens: int) -> torch.Tensor:
    """level_probs (B,K) over K nested levels -> soft per-token active prob (B,max_tokens).
    Token j is sent iff the chosen level's token-count > j (prefix)."""
    K = level_probs.shape[1]
    tok = torch.arange(max_tokens, device=level_probs.device)
    lt  = torch.tensor(level_tokens, device=level_probs.device, dtype=torch.long)  # (K,)
    include = (lt.view(K, 1) > tok.view(1, max_tokens)).to(level_probs.dtype)      # (K,max_tokens)
    return level_probs @ include                                                   # (B,max_tokens) in [0,1]


class TemporalRouter(nn.Module):
    """(z_t, z_{t-1}) -> distribution over K code-length levels, from the temporal
    difference. The scalar ||z_t - z_{t-1}||^2 is the validated signal; a small conv
    on the diff map adds spatial detail. Gumbel-softmax for a differentiable choice."""

    def __init__(self, C_vae: int, n_levels: int, d_hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(C_vae, 32, 3, 2, 1), nn.GELU(),   # 28->14
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),      # 14->7
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),      # (B,64)
        )
        self.head = nn.Sequential(nn.Linear(64 + 1, d_hidden), nn.GELU(),
                                  nn.Linear(d_hidden, n_levels))

    def forward(self, z_t, z_prev, tau: float = 1.0, hard: bool = False):
        diff   = z_t - z_prev
        scalar = (diff ** 2).mean(dim=(1, 2, 3), keepdim=True).flatten(1)   # (B,1) the validated signal
        logits = self.head(torch.cat([self.enc(diff), scalar], dim=-1))     # (B,K)
        if self.training:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            probs = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
        return probs, logits, scalar.squeeze(1)


class _TokenCoder(nn.Module):
    """z -> per-token (mu, log_var) for max_tokens tokens of token_dim symbols each."""

    def __init__(self, C_vae, H_vae, W_vae, max_tokens, token_dim, d_hidden=512):
        super().__init__()
        self.max_tokens, self.token_dim = max_tokens, token_dim
        D = max_tokens * token_dim
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(C_vae * H_vae * W_vae, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_hidden), nn.GELU())
        self.mu = nn.Linear(d_hidden, D)
        self.lv = nn.Linear(d_hidden, D)

    def forward(self, z, sample: bool = True):
        B = z.shape[0]
        h = self.net(z)
        mu = self.mu(h).view(B, self.max_tokens, self.token_dim)
        lv = self.lv(h).view(B, self.max_tokens, self.token_dim)
        s  = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        return mu, lv, s


class AdaptiveSemComSystem(nn.Module):
    """Adaptive-rate Wyner-Ziv SemCom: router + nested-prefix JSCC + fusion refiner.

    forward returns a dict with the reconstruction and the rate/length terms the
    trainer needs. The refiner is the DisagreementFusionRefiner (z̃ = z_hat + r,
    content from s̃ only); the routing mask makes calm frames send few tokens, so
    the channel is naturally quiet on calm (gate closed) and rich on disturbed.
    """

    def __init__(self, config: dict):
        super().__init__()
        C = config.get("C_vae", 4); H = config.get("H_vae", 28); W = config.get("W_vae", 28)
        self.max_tokens = config.get("jscc_max_tokens", 64)
        self.token_dim  = config.get("jscc_token_dim", 16)
        self.levels     = config.get("jscc_levels", [8, 16, 32, 64])   # nested token counts
        self.snr_db     = config["snr_db"]
        d_hidden = config.get("jscc_d_hidden", 512)
        grid = int(round(self.max_tokens ** 0.5))
        assert grid * grid == self.max_tokens, "max_tokens must be a perfect square"
        D_max = self.max_tokens * self.token_dim

        self.router  = TemporalRouter(C, len(self.levels))
        self.encoder = _TokenCoder(C, H, W, self.max_tokens, self.token_dim, d_hidden)
        self.prior   = _TokenCoder(C, H, W, self.max_tokens, self.token_dim, d_hidden)   # p(s̃|z_hat)
        self.refiner = DisagreementFusionRefiner(
            D_max, C, H, W,
            config.get("refine_d_model", 256), config.get("refine_n_heads", 8),
            config.get("refine_n_layers", 4), sig_grid=grid,
        )
        channel_type = config.get("channel_type", "rayleigh")
        self.channel = FadingChannel(self.snr_db, channel_type=channel_type, cdl_kwargs={
            "model": config.get("cdl_model", "C"), "delay_spread": config.get("cdl_delay_spread", 100e-9),
            "carrier_frequency": config.get("cdl_carrier_freq", 3.5e9), "max_speed": config.get("cdl_max_speed", 3.0)})

    def _tokens_to_flat(self, tok):                       # (B,T,dim) -> (B, dim*T) matching refiner reshape
        return tok.transpose(1, 2).reshape(tok.shape[0], -1)

    def forward(self, z_t, z_hat, z_prev, tau: float = 1.0, hard: bool = False,
                sample: bool = True, force_full: bool = False, withhold_zhat: bool = False):
        B = z_t.shape[0]
        # WITHHOLD z_hat from the refiner (base + context) so it MUST decode the channel —
        # prevents the refiner from leaning on the prior and the encoder from collapsing the
        # rate. z_hat still reaches self.prior (the WZ prior is unchanged). Used during warmup.
        z_ref = torch.zeros_like(z_hat) if withhold_zhat else z_hat
        probs, logits, tdiff = self.router(z_t, z_prev, tau=tau, hard=hard)      # (B,K)
        if force_full:
            # WARMUP: bypass the router, transmit ALL tokens so the encoder/refiner learn to
            # use the channel before the router is allowed to shorten codes (else it collapses
            # to the minimum length before the channel is ever useful).
            mask = torch.ones(B, self.max_tokens, device=z_t.device, dtype=z_t.dtype)
        else:
            mask = _prefix_mask(probs, self.levels, self.max_tokens)             # (B,T) soft active-prob per token

        mu_e, lv_e, s = self.encoder(z_t, sample=sample)                         # (B,T,dim)
        mu_p, lv_p, _ = self.prior(z_hat, sample=False)                          # Wyner-Ziv prior p(s̃|z_hat)

        s_masked = s * mask.unsqueeze(-1)                                        # drop non-transmitted tokens
        st_flat  = self.channel(self._tokens_to_flat(s_masked))                 # (B, dim*T)
        # attention padding mask for hard/eval routing (True = ignore token)
        sig_pad = (mask < 0.5) if (hard or not self.training) else None
        z_tilde = self.refiner._denoise_x0(None, None, z_ref, st_flat, sig_pad=sig_pad)

        # per-token Wyner-Ziv rate KL(q||p), q var = enc var + channel noise
        sig_n = 1.0 / (10.0 ** (self.snr_db / 10.0))
        q_var = lv_e.exp() + sig_n
        p_var = lv_p.exp().clamp(min=1e-8)
        kl_tok = 0.5 * (p_var.log() - q_var.log() + (q_var + (mu_e - mu_p).pow(2)) / p_var - 1.0).mean(-1)  # (B,T)
        rate   = (mask * kl_tok).sum(1)                                          # (B,) only sent tokens cost rate
        exp_tokens = mask.sum(1)                                                 # (B,) expected code length (tokens)
        return {"z_tilde": z_tilde, "rate": rate, "exp_tokens": exp_tokens,
                "probs": probs, "logits": logits, "tdiff": tdiff, "mask": mask}
