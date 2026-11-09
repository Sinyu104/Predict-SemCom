"""DVST — Deep Video Semantic Transmission (Wang et al., IEEE JSAC 41(1), 2023), adapted to VAE latents.

Temporal-context conditional JSCC with a learned entropy model that drives per-embedding
channel-bandwidth allocation. The "frame" is the VAE latent z_t (C_vae, 28, 28) rather than
pixels, so the whole pipeline is directly comparable to the Wyner-Ziv codec (same source,
same channel, same VLA read-out).

Paper -> code map (Fig. 1/2/4, Eq. 1-17):

    g_a(x_t | c~_t)      contextual analysis transform      DVST.g_a   -> y_t (B,Cy,H,W)
    f_e(y_t | d~_t)      contextual deep-JSCC encoder       DVST.f_e   -> s_t (B,Cs,H,W), masked by k_{t,i}
    W(.)                 AWGN / Rayleigh / CDL channel      MaskedChannel
    f_d(s^_t | d^_t)     contextual deep-JSCC decoder       DVST.f_d   -> y^_t
    g_s(y^_t | c^_t)     contextual synthesis transform     DVST.g_s   -> z^_t
    c~_t = phi_a(x^_{t-1})   feature-domain context          ContextGen (feature extractor + refinement)
    d~_t = gamma_e(y^_{t-1}) codeword-domain context         ContextGen (feature precoding + refinement)
    entropy model P(y|c, z, y_<i)  (hyperprior + AR + temporal prior, Laplace)   EntropyModel
    k_{t,i} = eta * -log2 P(y_{t,i})  (Eq. 10), quantised to the set V           DVST.allocate
    rate tokens added to the JSCC backbone input (variable-length transmission)  DVST.rate_tok

Deviations from the paper (all deliberate, documented in README.md):
  * No motion link / warping: contexts are formed from the previous frame directly (zero motion).
    A 28x28 latent grid is too coarse for optical flow; the previous-frame reference already
    carries the (static) background that the context is meant to supply.
  * Convolutional residual backbones instead of Swin blocks (28x28 grid; conv blocks are what the
    WZ codec uses too, so the capacity comparison is like-for-like).
  * Hyperprior latent z uses a per-channel factorised logistic density (not Balle's CDF MLP).
  * One embedding = one latent-grid cell (the paper uses 32x32-pixel patches; a latent cell is 8x8 px).
"""
from __future__ import annotations
import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = str(pathlib.Path(__file__).resolve().parents[2])
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from models import FadingChannel  # noqa: E402  (reused only for the per-symbol gain sampler)


# --------------------------------------------------------------------------- #
#  building blocks                                                            #
# --------------------------------------------------------------------------- #
class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.b(x)


def trunk(c_in, w, nb, c_out=None):
    layers = [nn.Conv2d(c_in, w, 3, 1, 1), nn.GELU()] + [Res(w) for _ in range(nb)]
    if c_out is not None:
        layers.append(nn.Conv2d(w, c_out, 1))
    return nn.Sequential(*layers)


class MaskedConv2d(nn.Conv2d):
    """Mask-A conv (PixelCNN): output at cell i sees only cells strictly before i in raster order."""

    def __init__(self, c_in, c_out, k=5):
        super().__init__(c_in, c_out, k, 1, k // 2)
        m = torch.ones_like(self.weight)
        m[:, :, k // 2, k // 2:] = 0
        m[:, :, k // 2 + 1:] = 0
        self.register_buffer("mask", m)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, 1, self.padding)


class ContextGen(nn.Module):
    """phi_a / gamma_e: extractor -> (warp: identity here) -> refinement, applied to the previous frame."""

    def __init__(self, c_in, c_out, w):
        super().__init__()
        self.extract = nn.Sequential(nn.Conv2d(c_in, w, 3, 1, 1), nn.GELU(), Res(w))
        self.refine = nn.Sequential(Res(w), nn.Conv2d(w, c_out, 3, 1, 1))

    def forward(self, x):
        return self.refine(self.extract(x))


def laplace_likelihood(y, mu, sigma):
    """P(y_bar) = integral of Laplace(mu, sigma) over [y-.5, y+.5]  (Eq. 8)."""
    sigma = sigma.clamp(min=0.05)

    def cdf(v):
        d = (v - mu) / sigma
        return 0.5 - 0.5 * d.sign() * torch.expm1(-d.abs())
    return (cdf(y + 0.5) - cdf(y - 0.5)).clamp(min=1e-9)


class EntropyModel(nn.Module):
    """Hyperprior + autoregressive spatial prior + temporal (context) prior, fused into Laplace(mu, sigma)."""

    def __init__(self, Cy, Cz, w, c_ctx):
        super().__init__()
        self.h_pe = nn.Sequential(nn.Conv2d(Cy, w, 3, 2, 1), nn.GELU(), nn.Conv2d(w, Cz, 3, 2, 1))      # 28 -> 7
        self.h_pd = nn.Sequential(nn.ConvTranspose2d(Cz, w, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(w, w, 4, 2, 1))
        self.h_ar = nn.Sequential(MaskedConv2d(Cy, w, 5), nn.GELU())
        self.h_tpe = nn.Sequential(nn.Conv2d(c_ctx, w, 3, 1, 1), nn.GELU(), nn.Conv2d(w, w, 3, 1, 1))
        self.h_pf = nn.Sequential(nn.Conv2d(3 * w, w, 1), nn.GELU(), nn.Conv2d(w, w, 1), nn.GELU(), nn.Conv2d(w, 2 * Cy, 1))
        self.z_loc = nn.Parameter(torch.zeros(1, Cz, 1, 1))          # factorised density for z
        self.z_logs = nn.Parameter(torch.zeros(1, Cz, 1, 1))
        self.Cy = Cy

    ste = False     # phase 2: straight-through rounding in training, so the allocation sees the SAME rounded y_bar as at test

    def quant(self, v, train):
        if train and not self.ste:
            return v + (torch.rand_like(v) - 0.5)          # uniform-noise proxy (phase 1, Balle et al.)
        r = torch.round(v)
        return v + (r - v).detach() if train else r

    def z_bits(self, zb):
        s = self.z_logs.exp()
        p = torch.sigmoid((zb + 0.5 - self.z_loc) / s) - torch.sigmoid((zb - 0.5 - self.z_loc) / s)
        return -torch.log2(p.clamp(min=1e-9)).sum(dim=(1, 2, 3))

    def forward(self, y, ctx):
        """y: (B,Cy,H,W) float. ctx: (B,c_ctx,H,W) temporal context. Returns (y_bar, bits_per_cell (B,1,H,W), z_bits (B,))."""
        tr = self.training
        yb = self.quant(y, tr)
        zb = self.quant(self.h_pe(y), tr)
        hp = self.h_pd(zb)
        out = self.h_pf(torch.cat([hp, self.h_ar(yb), self.h_tpe(ctx)], 1))
        mu, sg = out[:, :self.Cy], F.softplus(out[:, self.Cy:])
        bits = -torch.log2(laplace_likelihood(yb, mu, sg)).sum(1, keepdim=True)            # (B,1,H,W)  r_{t,i}
        return yb, bits, self.z_bits(zb)


class MaskedChannel(nn.Module):
    """AWGN / Rayleigh / CDL with coherent ZF, noise set by the power of the SENT symbols only.

    Same math as models.FadingChannel (consecutive reals paired into complex symbols, N0 = P/snr,
    x^ = y conj(h)/|h|^2) but the unit-power normalisation and the noise are computed over the
    entries the allocator actually transmits (mask==1); un-sent entries are zero and stay zero.
    """

    def __init__(self, snr_db=20.0, channel_type="awgn", cdl_kwargs=None):
        super().__init__()
        self.ch = FadingChannel(snr_db, channel_type, cdl_kwargs)

    @property
    def snr_db(self):
        return self.ch.snr_db

    @snr_db.setter
    def snr_db(self, v):
        self.ch.snr_db = v

    def forward(self, s, mask):
        B = s.shape[0]
        sf, mf = s.flatten(1), mask.flatten(1)
        if sf.shape[1] % 2:
            sf = F.pad(sf, (0, 1)); mf = F.pad(mf, (0, 1))
        n_sent = mf.sum(1, keepdim=True).clamp(min=1.0)
        p = (sf.pow(2) * mf).sum(1, keepdim=True) / n_sent                                  # power of sent entries
        sf = sf / p.clamp(min=1e-8).sqrt() * mf                                             # PowerNorm -> unit power
        snr_lin = 10.0 ** (self.ch.snr_db / 10.0)
        std = (1.0 / snr_lin) ** 0.5
        xr, xi = sf[:, 0::2], sf[:, 1::2]
        hr, hi = self.ch._sample_gains(B, xr.shape[1], s.device)
        yr = hr * xr - hi * xi + std * torch.randn_like(xr)
        yi = hr * xi + hi * xr + std * torch.randn_like(xi)
        m2 = (hr.pow(2) + hi.pow(2)).clamp(min=1e-8)
        out = torch.empty_like(sf)
        out[:, 0::2] = (yr * hr + yi * hi) / m2
        out[:, 1::2] = (-yr * hi + yi * hr) / m2
        out = out * mf
        return out[:, :s.flatten(1).shape[1]].view_as(s)


# --------------------------------------------------------------------------- #
#  DVST                                                                       #
# --------------------------------------------------------------------------- #
class DVST(nn.Module):
    """One P-frame codec. `ref_z` = previous (reconstructed) latent, `ref_y` = previous decoded semantic map."""

    def __init__(self, C_vae=4, Cy=32, Cs=16, W=96, nb=3, Cz=24, Wc=48,
                 levels=(0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16), eta_prime=1.0):
        super().__init__()
        assert max(levels) <= Cs and Cs % 2 == 0
        self.C_vae, self.Cy, self.Cs = C_vae, Cy, Cs
        self.register_buffer("V", torch.tensor(levels, dtype=torch.float32))          # allowed k_{t,i} values
        self.eta_prime = eta_prime
        self.alloc_snr = None          # if set, allocation uses this SNR's capacity (fixed rate) instead of the live SNR
        # contexts (Fig. 5): feature-domain c, codeword-domain d
        self.phi = ContextGen(C_vae, Wc, Wc)                    # tx/rx: from previous latent
        self.gam = ContextGen(Cy, Wc, Wc)                       # tx/rx: from previous decoded y^_{t-1}
        # nonlinear transform
        self.g_a = trunk(C_vae + Wc, W, nb, Cy)
        self.g_s = trunk(Cy + Wc, W, nb, C_vae)
        # deep JSCC with rate tokens
        self.rate_tok_e = nn.Embedding(len(levels), W)
        self.rate_tok_d = nn.Embedding(len(levels), W)
        self.fe_in = nn.Sequential(nn.Conv2d(Cy + Wc, W, 3, 1, 1), nn.GELU())
        self.fe_body = nn.Sequential(*[Res(W) for _ in range(nb)], nn.Conv2d(W, Cs, 1))
        self.fd_in = nn.Sequential(nn.Conv2d(Cs + Wc, W, 3, 1, 1), nn.GELU())
        self.fd_body = nn.Sequential(*[Res(W) for _ in range(nb)], nn.Conv2d(W, Cy, 1))
        self.entropy = EntropyModel(Cy, Cz, W, Wc)

    # ---- contexts: zeros => intra mode (no reference) ----
    def contexts(self, ref_z, ref_y):
        if ref_z is None:
            return None, None
        return self.phi(ref_z), self.gam(ref_y)

    def _ctx(self, x, c, d, ref_present_like):
        B, _, H, W = x.shape
        z = lambda ch: x.new_zeros(B, ch, H, W)
        return (c if c is not None else z(self.phi.refine[-1].out_channels),
                d if d is not None else z(self.gam.refine[-1].out_channels))

    # ---- rate allocation (Eq. 10) ----
    def allocate(self, bits, snr_db, lam_scale=1.0):
        """bits (B,1,H,W) -> k (reals per cell) quantised to V with straight-through, level index."""
        if self.alloc_snr is not None:
            snr_db = self.alloc_snr
        C = 0.5 * math.log2(1.0 + 10.0 ** (snr_db / 10.0))           # capacity, bits per REAL channel symbol
        k_cont = (self.eta_prime / C) * bits * lam_scale
        d = (k_cont.unsqueeze(-1) - self.V.view(1, 1, 1, 1, -1)).abs()
        idx = d.argmin(-1)                                            # (B,1,H,W)
        k_q = self.V[idx]
        k = k_cont + (k_q - k_cont).detach()                          # STE: gradient of the rate flows via k_cont
        return k, idx, k_q

    # ---- primary link ----
    def encode(self, z, c, d, snr_db):
        c, d = self._ctx(z, c, d, None)
        y = self.g_a(torch.cat([z, c], 1))
        yb, bits, zbits = self.entropy(y, c)
        k, idx, k_q = self.allocate(bits, snr_db)
        h = self.fe_in(torch.cat([y, d], 1)) + self.rate_tok_e(idx.squeeze(1)).permute(0, 3, 1, 2)
        s = self.fe_body(h)                                           # (B,Cs,H,W)
        keep = (torch.arange(self.Cs, device=z.device).view(1, -1, 1, 1) < k_q).float()
        return dict(y=y, s=s, keep=keep, k=k, k_q=k_q, idx=idx, bits=bits, zbits=zbits, yb=yb)

    def decode(self, s_hat, idx, c, d):
        c, d = self._ctx(s_hat, c, d, None)
        h = self.fd_in(torch.cat([s_hat, d], 1)) + self.rate_tok_d(idx.squeeze(1)).permute(0, 3, 1, 2)
        y_hat = self.fd_body(h)
        z_hat = self.g_s(torch.cat([y_hat, c], 1))
        return y_hat, z_hat

    def forward(self, z, ref_z, ref_y, channel, snr_db):
        """Full P-frame pass. ref_* None => intra. Returns recon, y_hat and the rate bookkeeping."""
        c, d = self.contexts(ref_z, ref_y)
        e = self.encode(z, c, d, snr_db)
        s_tx = e["s"] * e["keep"]
        s_hat = channel(s_tx, e["keep"])
        y_hat, z_hat = self.decode(s_hat, e["idx"], c, d)
        # transform-only branch (Eq. 21): g_s on the noisy-quantised y, no channel; regulariser / pretraining
        z_tr = self.g_s(torch.cat([e["yb"], self._ctx(z, c, d, None)[0]], 1))
        return dict(z_hat=z_hat, y_hat=y_hat, z_tr=z_tr, **e)

    @torch.no_grad()
    def intra_reference(self, z_prev):
        """Reference for a P-frame: the previous frame passed through the codec WITHOUT the channel and
        WITHOUT context (Eq. 2 / Fig. 2: the tx simulates the receiver; I-frame = intra-coded)."""
        c, d = self.contexts(None, None)
        e = self.encode(z_prev, c, d, 100.0)
        y_hat, z_hat = self.decode(e["s"] * e["keep"], e["idx"], c, d)
        return z_hat, y_hat
