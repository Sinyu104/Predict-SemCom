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
