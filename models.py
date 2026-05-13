"""
models.py  —  Neural network modules for the VAE-latent Wyner-Ziv SemCom System.

Transmitter (Robot):
    z_t  = VAE_encode(x_t)                           ← frozen SD-VAE
    s_t  ~ q(s_t | z_t) = N(μ_enc, σ²_enc)           ← JsccEncoder
    s̃_t  = RayleighChannel(s_t)

Receiver (Edge Server):
    ẑ_t  = CtrlWorld([z̃_{t-km},...,z̃_{t-1}], a_t)   ← frozen Stage-1 (ctrl_world_wrapper.py)
    p(s_t | ẑ_t) = N(μ_prior, σ²_prior)               ← SideInfoEncoder  (KL loss only)
    z̃_t  = RefinementDiffusion(ẑ_t, s̃_t)             ← SDEdit start from ẑ_t; s̃_t cross-attn
    x̃_t  = VAE_decode(z̃_t)                            ← frozen SD-VAE  (vae_wrapper.py)
    â_t  = Policy(x̃_t)

Modules in this file
--------------------
1. JsccEncoder          z_t          → μ_enc, log_var_enc, s_t   (B, D_jscc)
2. SideInfoEncoder      ẑ_t          → μ_prior, log_var_prior     (B, D_jscc)
3. RefinementDiffusion  (ẑ_t, s̃_t)  → z̃_t   — second diffusion, s̃_t via cross-attn
4. RayleighChannel      s_t          → s̃_t
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

    Mean-pools VAE latent over space → MLP → μ and log_var.
    No access to ẑ_t (Wyner-Ziv transmitter constraint).
    """

    def __init__(self, C_vae: int, H_vae: int, W_vae: int, D_jscc: int, d_hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(C_vae, d_hidden),
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
        h       = z.mean(dim=(-2, -1))
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
    Conditional prior: ẑ_t → p(s_t | ẑ_t) = N(μ_prior, σ²_prior)

    Used ONLY in the rate loss (KL term). No data flows through this module
    at inference. Same architecture as JsccEncoder.
    """

    def __init__(self, C_vae: int, H_vae: int, W_vae: int, D_jscc: int, d_hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(C_vae, d_hidden),
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
        h       = z_hat.mean(dim=(-2, -1))
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
    """Pre-LN self-attention + cross-attention (on s̃_t) + FFN block."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm3      = nn.LayerNorm(d_model)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N, d_model)  — patch tokens
        cond : (B, 1, d_model)  — received signal conditioning token
        """
        xn = self.norm1(x)
        h, _ = self.self_attn(xn, xn, xn)
        x = x + h

        xn = self.norm2(x)
        h, _ = self.cross_attn(xn, cond, cond)
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
        self.sig_proj    = nn.Linear(D_jscc, d_model)
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
    ) -> torch.Tensor:
        """Predict clean x_0 given x_{t''}, t'', ẑ_t, and s̃_t.

        Conditioning sequence fed to cross-attention:
            [ẑ_t spatial tokens (N) | s̃_t token (1)]  →  (B, N+1, d_model)

        ẑ_t tokens provide spatial prior; s̃_t token carries channel correction.
        """
        B, C, H, W = x_noisy.shape
        N = H * W
        self._init_pos(x_noisy.device, x_noisy.dtype)
        pos = self.pos_embed.to(x_noisy.dtype)                            # (1, N, d)

        # Noisy input tokens
        patches = x_noisy.reshape(B, C, N).transpose(1, 2)               # (B, N, C)
        x = self.patch_proj(patches) + pos                                # (B, N, d)

        # Timestep conditioning: broadcast to each token
        t_emb = self.t_proj(self.t_embed(t)).unsqueeze(1)                 # (B, 1, d)
        x     = x + t_emb

        # Conditioning sequence: ẑ_t spatial tokens + s̃_t global token
        zhat_patches = z_hat.reshape(B, C, N).transpose(1, 2)            # (B, N, C)
        zhat_tokens  = self.zhat_proj(zhat_patches) + pos                 # (B, N, d)
        sig_token    = self.sig_proj(s_tilde).unsqueeze(1)                # (B, 1, d)
        cond         = torch.cat([zhat_tokens, sig_token], dim=1)         # (B, N+1, d)

        for blk in self.blocks:
            x = blk(x, cond)
        x = self.norm(x)

        pred = self.out_proj(x)                                            # (B, N, C)
        return pred.transpose(1, 2).reshape(B, C, H, W)                   # (B, C, H, W)

    def forward_ddpm(
        self,
        z_t:         torch.Tensor,   # (B, C, H, W) — ground truth target
        z_hat:       torch.Tensor,   # (B, C, H, W) — Ctrl-World side info ẑ_t
        s_tilde:     torch.Tensor,   # (B, D_jscc)  — received signal s̃_t
        noise_level: int = 250,
    ) -> torch.Tensor:
        """
        True SDEdit training loss: noise z_t (not ẑ_t), condition on (ẑ_t, s̃_t).

        x_{t''} = sqrt(ᾱ_{t''}) z_t + sqrt(1-ᾱ_{t''}) ε,   t'' ~ U[1, noise_level]
        L        = MSE( denoiser(x_{t''}, t'', ẑ_t, s̃_t),  z_t )
        """
        B  = z_t.shape[0]
        t  = torch.randint(1, noise_level + 1, (B,), device=z_t.device)
        ab = self.alphas_cumprod[t].view(B, 1, 1, 1).float()
        eps     = torch.randn_like(z_t)
        x_noisy = ab.sqrt() * z_t + (1.0 - ab).sqrt() * eps              # noise z_t
        pred_x0 = self._denoise_x0(x_noisy, t, z_hat, s_tilde)
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
#  4. RAYLEIGH CHANNEL                                                        #
# ========================================================================== #

class RayleighChannel(nn.Module):
    """
    Flat Rayleigh fading channel: y = h·x + n,  h, n ~ CN(0,1)

    Coherent equalization divides by |h|² after reception.
    """

    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db

    @property
    def sigma_n_sq(self) -> float:
        return 1.0 / (10.0 ** (self.snr_db / 10.0))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, D_jscc) → s̃: (B, D_jscc)"""
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
        self.channel              = RayleighChannel(snr_db)

    @staticmethod
    def rate_loss(
        mu_enc:        torch.Tensor,
        log_var_enc:   torch.Tensor,
        mu_prior:      torch.Tensor,
        log_var_prior: torch.Tensor,
        snr_db:        float,
    ) -> torch.Tensor:
        """
        KL( q(s_t | z_t) || p(s_t | ẑ_t) )

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
        return kl.mean()
