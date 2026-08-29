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

    def __init__(self, C_vae: int, H_vae: int, W_vae: int, D_jscc: int, d_hidden: int = 512):
        super().__init__()
        self.in_dim = C_vae * H_vae * W_vae
        self.shared = nn.Sequential(
            nn.Linear(self.in_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(d_hidden, D_jscc)
        self.fc_log_var = nn.Linear(d_hidden, D_jscc)
        self.D_jscc = D_jscc

    def forward(
        self, z: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z : (B, C_vae, H_vae, W_vae)
        Returns μ_enc, log_var_enc, s_t — all (B, D_jscc)
        """
        h       = z.flatten(1)                      # (B, C_vae*H_vae*W_vae)
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

    def __init__(self, C_vae: int, H_vae: int, W_vae: int, D_jscc: int, d_hidden: int = 512):
        super().__init__()
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
    ):
        super().__init__()
        self.C_vae = C_vae
        self.H_vae = H_vae
        self.W_vae = W_vae

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
        sig          = self.sig_proj(
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
        """s: (B, D_jscc) → s̃: (B, D_jscc)"""
        B, D      = s.shape
        sig_power = s.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
        snr_lin   = 10.0 ** (self.snr_db / 10.0)
        noise_std = (sig_power / snr_lin).sqrt()

        h_re, h_im = self._sample_gains(B, D, s.device)

        n_re = noise_std * torch.randn_like(s) / math.sqrt(2)
        n_im = noise_std * torch.randn_like(s) / math.sqrt(2)
        y_re = h_re * s + n_re
        y_im = h_im * s + n_im

        h_mag_sq = (h_re.pow(2) + h_im.pow(2)).clamp(min=1e-8)
        return (y_re * h_re + y_im * h_im) / h_mag_sq


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

        self.jscc_encoder         = JsccEncoder(C_vae, H_vae, W_vae, D_jscc, d_hidden)
        self.side_info_encoder    = SideInfoEncoder(C_vae, H_vae, W_vae, D_jscc, d_hidden)
        self.refinement_diffusion = RefinementDiffusion(
            D_jscc, C_vae, H_vae, W_vae, refine_d, refine_h, refine_l
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
