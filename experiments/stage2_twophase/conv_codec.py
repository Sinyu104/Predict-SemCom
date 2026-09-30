"""Translation-equivariant JSCC codec for the change-masked allocation path.

The dense JsccEncoder (flatten -> MLP) MEMORIZES cube positions: it fits training
cubes to ~0.02 latent MSE but generalizes to only ~0.24 on held-out cubes (a 0.22
train/val gap), because a dense map has no translation equivariance -- to place a
sharp cube at position p it must have SEEN a cube at p, and the teleport positions
don't cover the held-out ones. Shrinking/regularizing the dense net makes val WORSE
(it just underfits); the fix is the right inductive bias, not fewer params.

A fully-convolutional codec "encodes a cube wherever it is" by construction, so the
train/val gap collapses (0.22 -> ~0.001) and it SCALES: held-out cube MSE
0.164 (W32) -> 0.110 (W64) -> 0.084 (W96) -> 0.075 (W128), vs the dense floor 0.24.

The channel wants a flat symbol vector, so the spatial code (B, Dch, H, W) is
flattened only to pass through FadingChannel, then reshaped back -- the conv weights
themselves stay equivariant. See project_jscc_codec_overfits.
"""
import torch
import torch.nn as nn


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.b(x)


class ConvCodec(nn.Module):
    """Fully-conv encoder/decoder with a spatial (B, Dch, H, W) channel bottleneck.

    W96/Dch16/nb3 is the validated sweet spot (held-out cube MSE 0.084, small gap).
    """

    def __init__(self, C_vae: int = 4, W: int = 96, Dch: int = 16, nb: int = 3):
        super().__init__()
        enc = [nn.Conv2d(C_vae, W, 3, 1, 1), nn.GELU()] + [_Res(W) for _ in range(nb)] + [nn.Conv2d(W, Dch, 1)]
        dec = [nn.Conv2d(Dch, W, 3, 1, 1), nn.GELU()] + [_Res(W) for _ in range(nb)] + [nn.Conv2d(W, C_vae, 3, 1, 1)]
        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(*dec)
        self.Dch = Dch

    def encode(self, x):
        return self.enc(x)                                 # (B, Dch, H, W)

    def forward(self, x, channel):
        c = self.enc(x)
        B, Dc, H, Wd = c.shape
        c = channel(c.flatten(1)).view(B, Dc, H, Wd)       # flatten only for the channel
        return self.dec(c)


def _conv_trunk(C_in, W, Dch, nb):
    body = [nn.Conv2d(C_in, W, 3, 1, 1), nn.GELU()] + [_Res(W) for _ in range(nb)]
    return nn.Sequential(*body), nn.Conv2d(W, Dch, 1), nn.Conv2d(W, Dch, 1)   # trunk, mu-head, logvar-head


class ConvWZCodec(nn.Module):
    """Wyner-Ziv conv codec: probabilistic encoder q(s|z_t), a side-info prior
    p(s|z_hat), and a decoder -- so the rate term KL(q||p) can be re-introduced on
    top of the conv codec (restores L = distortion + beta*KL).

      encoder (WZ-blind):  z_t*m -> (mu_e, logvar_e)   per code element (B,Dch,H,W)
      prior   (receiver):  z_hat -> (mu_p, logvar_p)   p(s~|z_hat), WZ-legal at the receiver
      channel:             s~ = channel(reparam(mu_e,logvar_e)),  q_var = sigma_e^2 + sigma_n^2
      decoder:             s~ (dropped channels filled with mu_p) -> rec

    KL is per element; summed per channel it gives the per-channel rate that concentrates
    on the unpredictable cube (q!=p) and vanishes on predictable regions (q~=p). See
    project_adaptive_rate_conv / project_jscc_codec_overfits.
    """

    def __init__(self, C_vae: int = 4, W: int = 96, Dch: int = 16, nb: int = 3):
        super().__init__()
        self.Dch = Dch
        self.enc_body, self.enc_mu, self.enc_lv = _conv_trunk(C_vae, W, Dch, nb)
        self.pri_body, self.pri_mu, self.pri_lv = _conv_trunk(C_vae, W, Dch, nb)
        dec = [nn.Conv2d(Dch, W, 3, 1, 1), nn.GELU()] + [_Res(W) for _ in range(nb)] + [nn.Conv2d(W, C_vae, 3, 1, 1)]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        h = self.enc_body(x)
        return self.enc_mu(h), self.enc_lv(h).clamp(-10.0, 2.0)     # (B,Dch,H,W) each

    def prior(self, z_hat):
        h = self.pri_body(z_hat)
        return self.pri_mu(h), self.pri_lv(h).clamp(-10.0, 2.0)

    @staticmethod
    def kl_elem(mu_e, lv_e, mu_p, lv_p, sig_n):
        """Per-element KL( N(mu_e, e^lv_e + sig_n) || N(mu_p, e^lv_p) )."""
        q_var = lv_e.exp() + sig_n
        p_var = lv_p.exp().clamp(min=1e-8)
        return 0.5 * (p_var.log() - q_var.log() + (q_var + (mu_e - mu_p).pow(2)) / p_var - 1.0)

    def decode(self, s):
        return self.dec(s)
