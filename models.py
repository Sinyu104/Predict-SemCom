"""
models.py  —  Neural network modules for the Predictive Semantic Communication System.

Module index
------------
1.  JsccEncoder      x → (mu, log_var, z)     Variational encoder
2.  Reshaper         z → y                     Task-oriented decoder
3.  Predictor        (z_{t-1}, a_{t-1}) → ẑ   GRU World Model / Digital Twin
4.  Subtractor       Δz = z − ẑ               Fixed math
5.  Sparsifier       Δz → sparse(Δz)           Fixed-τ or learned-τ
6.  Reconstructor    sparse → dense Δẑ         Fixed math (identity)
7.  RayleighChannel  Δz_sparse → received      Rayleigh fading + AWGN
8.  Adder            z_rec = ẑ + Δẑ           Fixed math
9.  PredictiveSemComSystem  Full pipeline

VIB Encoder design
-------------------
Following the ATROC paper, the Encoder outputs a Gaussian distribution
q_ϕ(z|x) = N(mu_ϕ(x), sigma_ϕ(x)^2 * I).

During training:    z is sampled via the reparameterisation trick.
During inference:   z = mu (no sampling — deterministic and reproducible).

The KL divergence KL(q_ϕ(z|x) || N(0,I)) is returned as a separate
term so the trainer can weight it independently (VIB beta1 term).

Gradient path for Stage 1
--------------------------
OpenVLA(Reshaper(Channel(Encoder(x)))) → task_loss
                ↑                ↑
        gradients flow      gradients flow
OpenVLA is frozen; only Encoder and Reshaper receive gradient updates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Strided residual conv block (encoder)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResBlockUp(nn.Module):
    """Upsampling residual block (decoder)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


# ========================================================================== #
#  1. JSCC ENCODER  (Variational, outputs distribution)                      #
# ========================================================================== #

class JsccEncoder(nn.Module):
    """
    Variational JSCC Encoder: x → (mu, log_var, z).

    Architecture
    ------------
    4 × stride-2 Conv blocks  →  Flatten  →  FC → (mu, log_var) each of
    size latent_dim.

    z is sampled via the reparameterisation trick during training so that
    gradients flow through z back to the convolutional backbone.
    During inference, z = mu (no randomness).

    Returns
    -------
    mu      : (B, latent_dim)   mean of the approximate posterior
    log_var : (B, latent_dim)   log-variance of the approximate posterior
    z       : (B, latent_dim)   sample (training) or mean (inference)
    """

    def __init__(self, obs_channels: int, obs_height: int,
                 obs_width: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            ResBlock(obs_channels, 32,  stride=2),
            ResBlock(32,           64,  stride=2),
            ResBlock(64,           128, stride=2),
            ResBlock(128,          256, stride=2),
        )

        feat_h   = obs_height // 16
        feat_w   = obs_width  // 16
        flat_dim = 256 * feat_h * feat_w

        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim * 2, latent_dim)

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x      : (B, C, H, W)
        sample : True  → reparameterisation trick  (training)
                 False → z = mu                    (inference)

        Returns
        -------
        mu, log_var, z  each (B, latent_dim)
        """
        h       = self.conv(x)
        shared  = self.fc_shared(h)
        mu      = self.fc_mu(shared)
        log_var = self.fc_log_var(shared).clamp(-10.0, 2.0)  # numerical stability

        if sample and self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z   = mu + eps * std
        else:
            z   = mu

        return mu, log_var, z

    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Analytical KL divergence: KL(N(mu, sigma^2) || N(0, I)).
        Returns the mean over the batch dimension.
        """
        return -0.5 * torch.mean(
            1.0 + log_var - mu.pow(2) - log_var.exp()
        )


# ========================================================================== #
#  2. RESHAPER  (Task-oriented decoder)                                      #
# ========================================================================== #

class Reshaper(nn.Module):
    """
    Task-oriented decoder: z → y.

    Unlike a reconstruction decoder that minimises pixel MSE, the Reshaper
    is trained to produce images y that maximise the frozen AI agent's
    task performance.  It is NOT trained to look good to humans.

    Architecture mirrors the JsccEncoder with ConvTranspose2d blocks.
    Output is normalised to [0,1] with Sigmoid.
    """

    def __init__(self, obs_channels: int, obs_height: int,
                 obs_width: int, latent_dim: int):
        super().__init__()
        self.feat_h   = obs_height // 16
        self.feat_w   = obs_width  // 16
        flat_dim      = 256 * self.feat_h * self.feat_w

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            ResBlockUp(256, 128),
            ResBlockUp(128, 64),
            ResBlockUp(64,  32),
            nn.ConvTranspose2d(32, obs_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → y: (B, C, H, W)"""
        h = self.fc(z).view(-1, 256, self.feat_h, self.feat_w)
        return self.deconv(h)


# ========================================================================== #
#  3. PREDICTOR  (World Model / Digital Twin)                                #
# ========================================================================== #

class Predictor(nn.Module):
    """
    GRU-based World Model — the Digital Twin of the robot.

    Runs identically on the Device and Edge Server with synchronised
    hidden states, producing the same prediction ẑ_t on both sides
    without any extra transmission.

    Input : (z_{t-1} ∥ a_{t-1}) ∈ ℝ^{latent_dim + action_dim}
    Output: ẑ_t ∈ ℝ^{latent_dim},  h_t (updated hidden state)
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size  = latent_dim + action_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        z_prev: torch.Tensor,               # (B, latent_dim)
        a_prev: torch.Tensor,               # (B, action_dim)
        h:      torch.Tensor | None = None, # (1, B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_pred : (B, latent_dim)
        h_next : (1, B, hidden_dim)
        """
        inp = torch.cat([z_prev, a_prev], dim=-1).unsqueeze(1)  # (B,1,D)
        out, h_next = self.gru(inp, h)
        return self.fc_out(out.squeeze(1)), h_next


# ========================================================================== #
#  4. SUBTRACTOR  (fixed math)                                               #
# ========================================================================== #

class Subtractor(nn.Module):
    """
    Innovation coding: Δz = z_t − ẑ_t

    Transmitting only Δz instead of z_t reduces bandwidth because a good
    Predictor (Digital Twin) makes Δz small.  Zero bandwidth cost when the
    twin predicts perfectly.  No learnable parameters.
    """

    def forward(
        self, z_t: torch.Tensor, z_pred: torch.Tensor
    ) -> torch.Tensor:
        return z_t - z_pred


# ========================================================================== #
#  5. SPARSIFIER                                                             #
# ========================================================================== #

class Sparsifier(nn.Module):
    """
    Suppress Δz components whose magnitude is below the threshold τ.

    Fixed mode  (tau_mode="fixed"):
        tau is a registered buffer (scalar broadcast to latent_dim).
        Component i is transmitted iff |Δz_i| > tau.
        No Stage-3 training needed.  Sweep tau for Globecom ablation.

    Learned mode  (tau_mode="learned"):
        tau is an nn.Parameter of shape (latent_dim,) trained in Stage-3
        with a soft-gate surrogate loss (differentiable approximation).

    Top-k budget cap  (optional, both modes):
        If top_k is set, only the k largest-magnitude components are
        transmitted, enforcing a fixed per-step bit budget.
    """

    def __init__(
        self,
        latent_dim: int,
        tau_mode:   str   = "fixed",
        tau_init:   float = 0.05,
        top_k:      int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.tau_mode   = tau_mode
        self.top_k      = top_k

        if tau_mode == "fixed":
            self.register_buffer(
                "tau", torch.full((latent_dim,), tau_init)
            )
        elif tau_mode == "learned":
            self.tau = nn.Parameter(
                torch.full((latent_dim,), tau_init)
            )
        else:
            raise ValueError(
                f"tau_mode must be 'fixed' or 'learned', got '{tau_mode}'"
            )

    def forward(
        self,
        delta_z:     torch.Tensor,
        temperature: float = 1.0,
        hard:        bool  = True,
    ) -> dict:
        """
        Parameters
        ----------
        delta_z     : (B, latent_dim)
        temperature : soft-gate temperature (learned mode, hard=False only)
        hard        : True  → hard 0/1 mask (inference / Stage 1 & 2)
                      False → soft gate ∈ (0,1) (Stage-3 training)

        Returns
        -------
        dict with keys:
          delta_z_sparse : (B, latent_dim)  zeroed at suppressed dims
          mask           : (B, latent_dim)  float mask
          rate           : scalar           mean fraction transmitted
          nnz            : scalar           avg non-zero count (detached)
        """
        tau = self.tau.clamp(min=0.0)

        if self.tau_mode == "fixed" or hard:
            mask = (delta_z.abs() > tau).float()
        else:
            mask = torch.sigmoid(
                (delta_z.abs() - tau) / max(temperature, 1e-6)
            )

        # Optional top-k budget cap
        if self.top_k is not None:
            k = min(self.top_k, self.latent_dim)
            topk_vals, _ = delta_z.abs().topk(k, dim=-1)
            kth_val      = topk_vals[:, -1].unsqueeze(-1)
            budget_mask  = (delta_z.abs() >= kth_val).float()
            mask         = mask * budget_mask

        delta_z_sparse = delta_z * mask
        rate           = mask.mean()
        nnz            = mask.detach().sum(dim=-1).mean()

        return {
            "delta_z_sparse": delta_z_sparse,
            "mask":           mask,
            "rate":           rate,
            "nnz":            nnz,
        }

    @staticmethod
    def bits_per_step(
        nnz: torch.Tensor, latent_dim: int, value_bits: int = 16
    ) -> torch.Tensor:
        """
        Estimated bits per timestep.

        Packet: [4 header bits] + nnz × (ceil(log2(latent_dim)) + value_bits)
        """
        index_bits = math.ceil(math.log2(max(latent_dim, 2)))
        return 4.0 + nnz * (index_bits + value_bits)


# ========================================================================== #
#  6. RECONSTRUCTOR  (fixed math — identity)                                 #
# ========================================================================== #

class Reconstructor(nn.Module):
    """
    Identity pass-through on the Edge Server.

    The received sparse vector already has zeros at suppressed positions.
    The Adder will use the Predictor's estimate at those positions, so no
    additional processing is needed here.  Kept as an explicit module for
    clarity and future extension (e.g. a learned denoiser).
    """

    def forward(self, received_sparse: torch.Tensor) -> torch.Tensor:
        return received_sparse


# ========================================================================== #
#  7. RAYLEIGH CHANNEL                                                       #
# ========================================================================== #

class RayleighChannel(nn.Module):
    """
    Flat Rayleigh fading channel with complex AWGN.

    Model:  y = h · x + n,   h, n ~ CN(0,1)

    Implemented in real arithmetic.  Coherent equalization divides by |h|².
    Zero components (suppressed by Sparsifier) remain zero after the channel
    so the Adder correctly falls back to the Predictor estimate at those dims.

    Parameters
    ----------
    snr_db : float  Signal-to-Noise Ratio in dB
    """

    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, delta_z_sparse: torch.Tensor) -> torch.Tensor:
        """
        delta_z_sparse : (B, latent_dim)  sparse innovation
        →  received    : (B, latent_dim)  equalized, zero-preserved
        """
        nonzero_mask = (delta_z_sparse != 0).float()
        n_nonzero    = nonzero_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sig_power    = (
            delta_z_sparse.pow(2).sum(dim=-1, keepdim=True) / n_nonzero
        ).clamp(min=1e-8)

        snr_lin  = 10.0 ** (self.snr_db / 10.0)
        noise_std = (sig_power / snr_lin).sqrt()    # (B, 1)

        # h ~ CN(0,1): h_re, h_im ~ N(0, 1/√2)
        h_re = torch.randn(delta_z_sparse.size(0), 1,
                           device=delta_z_sparse.device) / math.sqrt(2)
        h_im = torch.randn(delta_z_sparse.size(0), 1,
                           device=delta_z_sparse.device) / math.sqrt(2)

        y_re = h_re * delta_z_sparse
        y_im = h_im * delta_z_sparse
        n_re = noise_std * torch.randn_like(delta_z_sparse) / math.sqrt(2)
        n_im = noise_std * torch.randn_like(delta_z_sparse) / math.sqrt(2)

        h_mag_sq = (h_re.pow(2) + h_im.pow(2)).clamp(min=1e-8)
        received = ((y_re + n_re) * h_re + (y_im + n_im) * h_im) / h_mag_sq

        # Re-apply zero mask so suppressed dims stay zero
        return received * nonzero_mask


# ========================================================================== #
#  8. ADDER  (fixed math)                                                    #
# ========================================================================== #

class Adder(nn.Module):
    """
    Latent recovery on the Edge Server:  z_rec = ẑ_t + received

    Transmitted dims:  z_rec_i ≈ z_t_i  (up to channel noise)
    Suppressed dims:   z_rec_i = ẑ_t_i  (Predictor estimate, no correction)

    No learnable parameters.
    """

    def forward(
        self, z_pred: torch.Tensor, received: torch.Tensor
    ) -> torch.Tensor:
        return z_pred + received


# ========================================================================== #
#  9. FULL SYSTEM                                                            #
# ========================================================================== #

class PredictiveSemComSystem(nn.Module):
    """
    End-to-end Predictive Semantic Communication System.

    Data flow (one time step t)
    ---------------------------
    DEVICE (Physical Robot):
      x_t ──[JsccEncoder]──► (mu, log_var, z_t)
      (z_{t-1}, a_{t-1}) ──[Predictor]──► ẑ_t        ← Digital Twin sync
      z_t - ẑ_t ──[Subtractor]──► Δz_t               ← Twin gap
      Δz_t ──[Sparsifier]──► Δz_sparse                ← Drop small dims
      Δz_sparse ──[Channel]──► received               ← Wireless link

    EDGE SERVER (Digital Twin):
      (z_{t-1}, a_{t-1}) ──[Predictor]──► ẑ_t         ← Free (same weights)
      received ──[Reconstructor]──► Δẑ
      ẑ_t + Δẑ ──[Adder]──► z_rec
      z_rec ──[Reshaper]──► y_rec
      y_rec ──[OpenVLAAgent]──► a_t

    The OpenVLAAgent is stored as a plain attribute (not nn.Module child)
    so its parameters never appear in self.parameters() and cannot
    accidentally be passed to an optimiser.

    Parameters
    ----------
    config   : dict from config.py
    agent    : OpenVLAAgent or OpenVLAStub instance (required)
    use_stub : bool  convenience flag — if True, ignores agent argument
                     and builds an OpenVLAStub internally
    """

    def __init__(self, config: dict, agent=None, use_stub: bool = False):
        super().__init__()

        C  = config["obs_channels"]
        H  = config["obs_height"]
        W  = config["obs_width"]
        ld = config["latent_dim"]
        hd = config["hidden_dim"]
        ad = config["action_dim"]

        self.jscc_encoder  = JsccEncoder(C, H, W, ld)
        self.reshaper      = Reshaper(C, H, W, ld)
        self.predictor     = Predictor(ld, ad, hd)
        self.subtractor    = Subtractor()
        self.sparsifier    = Sparsifier(
            latent_dim = ld,
            tau_mode   = config.get("tau_mode", "fixed"),
            tau_init   = config.get("tau",      0.05),
            top_k      = config.get("top_k",    None),
        )
        self.channel       = RayleighChannel(config["snr_db"])
        self.reconstructor = Reconstructor()
        self.adder         = Adder()

        # Store agent as a plain attribute so its 7B weights are never
        # included in self.parameters().
        if use_stub or agent is None:
            from openvla_agent import OpenVLAStub
            self.agent = OpenVLAStub(
                obs_channels = C,
                obs_height   = H,
                obs_width    = W,
                action_dim   = ad,
            )
        else:
            self.agent = agent

        self.latent_dim = ld
        self.tau_mode   = config.get("tau_mode", "fixed")

    def forward(
        self,
        x_t:              torch.Tensor,
        z_prev:           torch.Tensor,
        a_prev:           torch.Tensor,
        h:                torch.Tensor | None = None,
        bypass_predictor: bool  = False,
        sparsify:         bool  = True,
        sample_z:         bool  = True,
        temperature:      float = 1.0,
        hard:             bool  = True,
    ) -> dict:
        """
        Single time-step forward pass.

        Parameters
        ----------
        x_t              : (B, C, H, W)
        z_prev           : (B, latent_dim)
        a_prev           : (B, action_dim)
        h                : GRU hidden state | None
        bypass_predictor : True during Stage 1 (z_pred = 0)
        sparsify         : False during Stage 1 & 2 (transmit full Δz)
        sample_z         : True during training (reparameterisation)
        temperature      : soft-gate temperature (Stage 3)
        hard             : True except during Stage 3 soft-gate training

        Returns
        -------
        dict with keys:
          mu, log_var, z_t, z_pred, delta_z, delta_z_sparse,
          mask, rate, nnz, bits_per_step,
          z_rec, y_rec, h_next
        """
        # ── Encode ────────────────────────────────────────────────────── #
        mu, log_var, z_t = self.jscc_encoder(x_t, sample=sample_z)

        # ── Predict (Digital Twin) ────────────────────────────────────── #
        if bypass_predictor:
            z_pred = torch.zeros_like(z_t)
            h_next = h
        else:
            z_pred, h_next = self.predictor(z_prev, a_prev, h)

        # ── Innovation ───────────────────────────────────────────────── #
        delta_z = self.subtractor(z_t, z_pred)

        # ── Sparsify ─────────────────────────────────────────────────── #
        if sparsify:
            sp = self.sparsifier(delta_z, temperature=temperature, hard=hard)
        else:
            ones = torch.ones_like(delta_z)
            sp   = {
                "delta_z_sparse": delta_z,
                "mask":           ones,
                "rate":           torch.tensor(1.0, device=delta_z.device),
                "nnz":            torch.tensor(
                    float(self.latent_dim), device=delta_z.device
                ),
            }

        # ── Channel ───────────────────────────────────────────────────── #
        received    = self.channel(sp["delta_z_sparse"])

        # ── Recover latent ────────────────────────────────────────────── #
        delta_z_hat = self.reconstructor(received)
        z_rec       = self.adder(z_pred, delta_z_hat)

        # ── Decode ───────────────────────────────────────────────────── #
        y_rec = self.reshaper(z_rec)

        # ── Bit-rate ─────────────────────────────────────────────────── #
        bits = Sparsifier.bits_per_step(sp["nnz"], self.latent_dim)

        return {
            "mu":              mu,
            "log_var":         log_var,
            "z_t":             z_t,
            "z_pred":          z_pred,
            "delta_z":         delta_z,
            "delta_z_sparse":  sp["delta_z_sparse"],
            "mask":            sp["mask"],
            "rate":            sp["rate"],
            "nnz":             sp["nnz"],
            "bits_per_step":   bits,
            "z_rec":           z_rec,
            "y_rec":           y_rec,
            "h_next":          h_next,
        }
