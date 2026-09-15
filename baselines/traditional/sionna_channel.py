"""
NVIDIA Sionna channel for the traditional SSCC baseline (Sionna >= 2.0, torch).

Coded-modulation chain:

    info bits ─▶ 5G-LDPC encode ─▶ QAM map ─▶ [ per-symbol fading h + AWGN ] ─▶
                ZF equalize (x̂ = y/h) ─▶ QAM demap (LLR) ─▶ 5G-LDPC decode

Three channel models share the exact same receiver (coherent zero-forcing +
per-symbol effective noise N0/|h|²), so only the per-symbol complex gain `h`
differs:

    awgn      h = 1                          (rate-distortion reference)
    rayleigh  h ~ CN(0,1) i.i.d. per symbol  (flat fading; matches
                                              models.RayleighChannel exactly)
    cdl       h from a Sionna 3GPP TR-38.901 CDL model, evaluated as an OFDM
              frequency response (frequency- and time-selective fading)

The unit-energy constellation and normalized channels (E|h|² = 1) give
N0 = 1/snr_lin for every model, matching models.RayleighChannel and keeping the
SNR axis comparable across channels.

Sionna 2.0 is PyTorch-based and defaults to CUDA; everything here stays on
`device` and only the recovered bits are returned to the host.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .bit_utils import bytes_to_bits, bits_to_bytes  # re-exported for convenience

# ── Sionna 2.0 (sionna.phy, torch backend) ──────────────────────────────── #
try:
    import sionna
    from sionna.phy import config as _sn_config
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.mapping import Mapper, Demapper
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "NVIDIA Sionna >= 2.0 is required for the channel.\n"
        "    pip install sionna\n"
        f"(original error: {exc})"
    ) from exc


CHANNELS = ("awgn", "rayleigh", "cdl")


# ========================================================================== #
#  CDL frequency-response generator (Sionna TR-38.901)                       #
# ========================================================================== #
class _CDLGains:
    """
    Lazily builds a Sionna CDL model + OFDM resource grid and yields flattened
    per-subcarrier complex channel gains.  Each realization covers
    num_ofdm_symbols * fft_size gains; we draw as many realizations as needed
    to fill the requested symbol count.
    """

    def __init__(
        self,
        model: str = "A",
        delay_spread: float = 100e-9,
        carrier_frequency: float = 3.5e9,
        min_speed: float = 0.0,
        max_speed: float = 3.0,
        fft_size: int = 128,
        num_ofdm_symbols: int = 14,
        subcarrier_spacing: float = 30e3,
        device: str | None = None,
    ):
        from sionna.phy.channel.tr38901 import CDL, AntennaArray
        from sionna.phy.channel import GenerateOFDMChannel
        from sionna.phy.ofdm import ResourceGrid

        arr = AntennaArray(
            num_rows=1, num_cols=1, polarization="single",
            polarization_type="V", antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
        cdl = CDL(
            model=model, delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            ut_array=arr, bs_array=arr, direction="downlink",
            min_speed=min_speed, max_speed=max_speed,
        )
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=fft_size // 8,
        )
        # normalize so E|h|² = 1 -> SNR convention matches the other channels.
        self._gen = GenerateOFDMChannel(cdl, rg, normalize_channel=True)
        self._per_real = num_ofdm_symbols * fft_size
        self.device = device

    def __call__(self, num_symbols: int) -> torch.Tensor:
        n_real = math.ceil(num_symbols / self._per_real)
        h = self._gen(batch_size=n_real).reshape(-1)[:num_symbols]
        if self.device is not None:
            h = h.to(self.device)
        return h


# ========================================================================== #
#  Coded channel                                                             #
# ========================================================================== #
class SionnaCodedChannel:
    """
    5G-LDPC + QAM over a selectable Sionna channel.

    Args:
        k, n:  LDPC information / codeword length (rate = k/n).
        num_bits_per_symbol: QAM order exponent (2=QPSK, 4=16-QAM, 6=64-QAM);
                             n must be divisible by it.
        snr_db: per-symbol Es/N0 in dB (N0 = 1/snr_lin).
        channel: "awgn" | "rayleigh" | "cdl".
        num_iter: LDPC belief-propagation iterations.
        device: torch device ("cuda"/"cpu"); default = Sionna's config.device.
        seed: RNG seed for reproducible fading/noise.
        cdl_kwargs: forwarded to the CDL generator (model, delay_spread,
                    carrier_frequency, min_speed, max_speed, ...).
    """

    def __init__(
        self,
        k: int = 3072,
        n: int = 6144,
        num_bits_per_symbol: int = 2,
        snr_db: float = 10.0,
        channel: str = "rayleigh",
        num_iter: int = 20,
        device: str | None = None,
        seed: int | None = 0,
        **cdl_kwargs,
    ):
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got '{channel}'.")
        if n % num_bits_per_symbol != 0:
            raise ValueError(
                f"n ({n}) must be divisible by num_bits_per_symbol "
                f"({num_bits_per_symbol})."
            )
        self.k = int(k)
        self.n = int(n)
        self.nbps = int(num_bits_per_symbol)
        self.snr_db = float(snr_db)
        self.channel = channel
        self.device = device or str(getattr(_sn_config, "device", "cpu"))
        if seed is not None:
            _sn_config.seed = seed
            torch.manual_seed(seed)

        self.encoder  = LDPC5GEncoder(self.k, self.n)
        self.decoder  = LDPC5GDecoder(self.encoder, hard_out=True,
                                      return_infobits=True, num_iter=int(num_iter))
        self.mapper   = Mapper("qam", num_bits_per_symbol=self.nbps)
        self.demapper = Demapper("app", "qam", num_bits_per_symbol=self.nbps)

        self._cdl = None
        if channel == "cdl":
            self._cdl = _CDLGains(device=self.device, **cdl_kwargs)

    # -- properties ------------------------------------------------------- #
    @property
    def code_rate(self) -> float:
        return self.k / self.n

    @property
    def no(self) -> float:
        """Noise power N0 = 1 / snr_lin (matches models.RayleighChannel)."""
        return 10.0 ** (-self.snr_db / 10.0)

    # -- fading gains ----------------------------------------------------- #
    def _gains(self, num_symbols: int) -> torch.Tensor:
        dev = self.device
        if self.channel == "awgn":
            return torch.ones(num_symbols, dtype=torch.complex64, device=dev)
        if self.channel == "rayleigh":
            re = torch.randn(num_symbols, device=dev)
            im = torch.randn(num_symbols, device=dev)
            return torch.complex(re, im) / math.sqrt(2.0)
        return self._cdl(num_symbols).to(dev)          # cdl

    # -- main entry point ------------------------------------------------- #
    def transmit(self, info_bits: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Push a 0/1 bit stream through the coded channel.

        Returns (recovered_bits, stats) with recovered_bits the same length as
        info_bits and stats = {'ber', 'num_blocks', 'code_rate', 'snr_db',
        'channel'}.  The post-LDPC BER is what makes or breaks the video
        decoder downstream.
        """
        info_bits = np.asarray(info_bits, dtype=np.uint8).ravel()
        L = info_bits.size
        pad = (-L) % self.k
        padded = np.concatenate([info_bits, np.zeros(pad, np.uint8)])
        blocks = padded.reshape(-1, self.k)                       # [B, k]
        num_blocks = blocks.shape[0]

        u = torch.from_numpy(blocks).float().to(self.device)
        c = self.encoder(u)                                        # [B, n]
        x = self.mapper(c)                                         # [B, n/nbps] cplx

        num_sym = x.shape[1]
        xs = x.reshape(-1)                                         # [B*num_sym]
        h  = self._gains(xs.numel()).to(xs.dtype)

        no = self.no
        noise = math.sqrt(no / 2.0) * torch.complex(
            torch.randn(xs.numel(), device=self.device),
            torch.randn(xs.numel(), device=self.device),
        ).to(xs.dtype)
        y = h * xs + noise

        # Coherent zero-forcing equalization; post-eq noise is coloured.
        x_hat  = (y / h).reshape(num_blocks, num_sym)
        no_eff = (no / h.abs().clamp(min=1e-12).pow(2)).reshape(num_blocks, num_sym)

        llr   = self.demapper(x_hat, no_eff)                       # [B, n]
        u_hat = self.decoder(llr)                                  # [B, k] hard

        out = u_hat.round().to(torch.uint8).cpu().numpy().reshape(-1)[:L]
        ber = float(np.mean(out != info_bits)) if L else 0.0
        stats = {
            "ber":        ber,
            "num_blocks": int(num_blocks),
            "code_rate":  self.code_rate,
            "snr_db":     self.snr_db,
            "channel":    self.channel,
        }
        return out, stats
