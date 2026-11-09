"""Digital (uncoded) transmission of the WZ code: uniform scalar quantizer -> Gray bits -> QAM -> fading+AWGN -> ZF -> hard demap.

Same modulation chain as the SSCC baseline (baselines/traditional/sionna_channel.py: Sionna QAM Mapper/Demapper, unit-energy
constellation, N0 = 1/snr_lin, awgn / rayleigh / CDL gains, coherent ZF) but NO channel coding and NO packet dropping:
every bit is delivered, wrong bits included (hard decision), so a bit error just perturbs the dequantized value.

Drop-in for FadingChannel: forward(s: (B, D) reals) -> (B, D) reals.  D = n_ch * group, where `group` = cells per code channel
(28*56); each code channel gets its own range (max |value|, sent as an error-free 16-bit header per channel and frame).

    ch = DigitalQAMChannel(snr_db=20, bits=4, nbps=4, channel_type="cdl", cdl_kwargs={...}, group=28*56)
    s_rx = ch(s)          # ch.last = {"ber", "bits", "symbols", "header_bits"}
"""
import math
import torch

from baselines.traditional.sionna_channel import _CDLGains
from sionna.phy.mapping import Mapper, Demapper


class DigitalQAMChannel:
    def __init__(self, snr_db=20.0, bits=4, nbps=4, channel_type="awgn", cdl_kwargs=None, group=28 * 56, device="cuda"):
        assert channel_type in ("awgn", "rayleigh", "cdl")
        self.snr_db, self.bits, self.nbps, self.channel_type, self.group, self.device = float(snr_db), int(bits), int(nbps), channel_type, int(group), device
        self.mapper = Mapper("qam", num_bits_per_symbol=self.nbps)
        self.demapper = Demapper("app", "qam", num_bits_per_symbol=self.nbps, hard_out=True)
        self._cdl = _CDLGains(device=device, **(cdl_kwargs or {})) if channel_type == "cdl" else None
        self.last = {}

    @property
    def no(self):
        return 10.0 ** (-self.snr_db / 10.0)

    def _gains(self, B, M, dev):
        if self.channel_type == "awgn":
            return torch.ones(B, M, dtype=torch.complex64, device=dev)
        if self.channel_type == "rayleigh":
            return torch.complex(torch.randn(B, M, device=dev), torch.randn(B, M, device=dev)) / math.sqrt(2.0)
        return self._cdl(B * M).reshape(B, M).to(dev)

    # -- transmitter: reals -> bits ----------------------------------------------------------------- #
    def quantize_bits(self, s):
        B, D = s.shape; b = self.bits; L = 2 ** b
        x = s.reshape(B, D // self.group, self.group)
        scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-8)                     # per frame and code channel
        idx = torch.round((x / scale + 1.0) * 0.5 * (L - 1)).long().clamp(0, L - 1)
        gray = idx ^ (idx >> 1)                                                     # Gray: neighbouring levels differ by 1 bit
        bits = ((gray.unsqueeze(-1) >> torch.arange(b - 1, -1, -1, device=s.device)) & 1).reshape(B, -1)
        return bits, scale

    # -- receiver: bits -> reals -------------------------------------------------------------------- #
    def bits_to_values(self, bits, scale, shape):
        B, D = shape; b = self.bits; L = 2 ** b
        gray = (bits.reshape(B, D, b).long() << torch.arange(b - 1, -1, -1, device=bits.device)).sum(-1)
        idx = gray.clone()
        for sh in (1, 2, 4, 8):                                                     # Gray -> binary
            idx = idx ^ (idx >> sh)
        x = (idx.float() / (L - 1) * 2.0 - 1.0).reshape(B, D // self.group, self.group)
        return (x * scale).reshape(B, D)

    @torch.no_grad()
    def __call__(self, s):
        dev = s.device; B, D = s.shape
        bits, scale = self.quantize_bits(s)
        nb = bits.shape[1]; pad = (-nb) % self.nbps
        tx = torch.cat([bits, torch.zeros(B, pad, dtype=bits.dtype, device=dev)], 1) if pad else bits
        x = self.mapper(tx.float())                                                 # (B, M) complex, unit energy
        M = x.shape[1]
        h = self._gains(B, M, dev).to(x.dtype)
        n = math.sqrt(self.no / 2.0) * torch.complex(torch.randn(B, M, device=dev), torch.randn(B, M, device=dev)).to(x.dtype)
        xh = (x * h + n) / h                                                        # coherent ZF
        ne = self.no / h.abs().clamp(min=1e-12).pow(2)
        rx = self.demapper(xh, ne).round().long()[:, :nb]                           # hard bits; no FEC, nothing dropped
        self.last = {"ber": float((rx != bits).float().mean()), "bits": int(nb), "symbols": int(M), "header_bits": 16 * (D // self.group)}
        return self.bits_to_values(rx, scale, (B, D))


class DigitalSDMCMChannel(DigitalQAMChannel):
    """Same uncoded link, but with the sDMCM idea (Teng et al., IEEE IoT-J 2025) in place of Gray labelling:
    the quantiser index stays NATURAL binary, each QAM axis is a PAM whose levels are in numerical order
    (so a nearest-neighbour symbol error changes the value by one PAM step, not by a bit-weight jump), and the bits of a value are
    sent MSB-first, m = nbps/2 bits per PAM symbol (n = bits must be divisible by m).  I/Q of one QAM symbol take two consecutive PAM symbols.
    No redundancy, no coding, nothing dropped.  nbps=2 degenerates to QPSK with a natural-binary quantiser."""
    def __init__(self, snr_db=20.0, bits=8, nbps=4, channel_type="awgn", cdl_kwargs=None, group=28 * 56, device="cuda"):
        assert nbps % 2 == 0 and bits % (nbps // 2) == 0, "need n % m == 0 (m = bits per PAM axis)"
        self.m = nbps // 2; self.Lp = 2 ** self.m
        self.snr_db, self.bits, self.nbps, self.channel_type, self.group, self.device = float(snr_db), int(bits), int(nbps), channel_type, int(group), device
        self._cdl = _CDLGains(device=device, **(cdl_kwargs or {})) if channel_type == "cdl" else None
        self.norm = math.sqrt(2.0 * (4 ** self.m - 1) / 3.0)             # unit average symbol energy (levels -(L-1)..(L-1) step 2)
        self.last = {}

    def quantize_bits(self, s):                                           # natural binary, MSB first
        B, D = s.shape; b = self.bits; L = 2 ** b
        x = s.reshape(B, D // self.group, self.group)
        scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-8)
        idx = torch.round((x / scale + 1.0) * 0.5 * (L - 1)).long().clamp(0, L - 1)
        return ((idx.unsqueeze(-1) >> torch.arange(b - 1, -1, -1, device=s.device)) & 1).reshape(B, -1), scale

    def bits_to_values(self, bits, scale, shape):
        B, D = shape; b = self.bits; L = 2 ** b
        idx = (bits.reshape(B, D, b).long() << torch.arange(b - 1, -1, -1, device=bits.device)).sum(-1)
        return ((idx.float() / (L - 1) * 2.0 - 1.0).reshape(B, D // self.group, self.group) * scale).reshape(B, D)

    @torch.no_grad()
    def __call__(self, s):
        dev = s.device; B, D = s.shape; m, Lp = self.m, self.Lp
        bits, scale = self.quantize_bits(s); nb = bits.shape[1]
        pad = (-nb) % (2 * m)
        tx = torch.cat([bits, torch.zeros(B, pad, dtype=bits.dtype, device=dev)], 1) if pad else bits
        lvl = (tx.reshape(B, -1, m).long() << torch.arange(m - 1, -1, -1, device=dev)).sum(-1)       # PAM index, natural order
        amp = (2.0 * lvl.float() - (Lp - 1)) / self.norm                                               # PAM amplitude
        x = torch.complex(amp[:, 0::2], amp[:, 1::2]); M = x.shape[1]
        h = self._gains(B, M, dev).to(x.dtype)
        n = math.sqrt(self.no / 2.0) * torch.complex(torch.randn(B, M, device=dev), torch.randn(B, M, device=dev)).to(x.dtype)
        xh = (x * h + n) / h                                                                           # coherent ZF
        r = torch.stack([xh.real, xh.imag], -1).reshape(B, -1) * self.norm                             # back to integer-spaced levels
        rl = torch.round((r + (Lp - 1)) / 2.0).long().clamp(0, Lp - 1)                                 # nearest PAM level
        rx = ((rl.unsqueeze(-1) >> torch.arange(m - 1, -1, -1, device=dev)) & 1).reshape(B, -1)[:, :nb]
        self.last = {"ber": float((rx != bits).float().mean()), "bits": int(nb), "symbols": int(M), "header_bits": 16 * (D // self.group)}
        return self.bits_to_values(rx, scale, (B, D))
