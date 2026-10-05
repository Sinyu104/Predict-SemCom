"""
Per-frame coded link for the streaming SSCC baseline.

A frame's compressed bytes + CRC32 go through 5G-LDPC (codeword sized to the frame,
split only above 8448 info bits), QAM, and a Sionna channel (awgn / rayleigh / 3GPP CDL),
with the same per-symbol coherent zero-forcing receiver and N0 = 1/snr_lin convention as
models.FadingChannel.  A frame whose CRC fails is reported as LOST (None).

Link adaptation: given the channel and SNR, pick the highest-throughput MCS
(QAM order, LDPC rate) from a calibrated table whose BLER <= target (see
calibrate_mcs.py -> mcs_table.json).  If no MCS meets the target, use the most robust one.

Bandwidth accounting: `info["n_symbols"]` = complex channel uses spent on this frame.
(Genie: the receiver knows the frame length, as in the existing baseline; a real header would
be protected separately and is negligible next to the payload.)
"""
from __future__ import annotations

import json
import math
import os
import zlib

import numpy as np
import torch

from .bit_utils import bytes_to_bits, bits_to_bytes
from .sionna_channel import _CDLGains, _sn_config, LDPC5GEncoder, LDPC5GDecoder, Mapper, Demapper

KMAX = 8448          # 5G LDPC base-graph-1 maximum information length

# (bits per QAM symbol, LDPC rate).  Ordered by throughput.
MCS_TABLE = [(2, 0.35), (2, 1 / 2), (2, 3 / 4), (4, 1 / 2), (4, 3 / 4), (6, 2 / 3), (6, 3 / 4)]


def mcs_name(m):
    return f"{2 ** m[0]}QAM-r{m[1]:.2f}"


def _crc_append(payload: bytes) -> bytes:
    return payload + zlib.crc32(payload).to_bytes(4, "big")


def _crc_check(data: bytes):
    if len(data) < 4:
        return None
    body, crc = data[:-4], data[-4:]
    return body if zlib.crc32(body).to_bytes(4, "big") == crc else None


class AdaptiveLink:
    def __init__(self, channel: str = "awgn", snr_db: float = 10.0, mcs: tuple | None = None,
                 table_path: str | None = None, bler_target: float = 1e-2, num_iter: int = 20,
                 device: str | None = None, seed: int | None = None, cdl_model: str = "C",
                 cdl_kwargs: dict | None = None):
        self.channel = channel
        self.snr_db = float(snr_db)
        self.num_iter = num_iter
        self.device = device or str(getattr(_sn_config, "device", "cpu"))
        if seed is not None:
            _sn_config.seed = seed
            torch.manual_seed(seed)
        self._cdl = None
        if channel == "cdl":
            kw = {"model": cdl_model, "delay_spread": 100e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}
            kw.update(cdl_kwargs or {})
            self._cdl = _CDLGains(device=self.device, **kw)
        self._enc, self._dec, self._map, self._dem = {}, {}, {}, {}
        self.mcs = mcs or self.select_mcs(table_path, bler_target)
        self.stats = {"frames": 0, "lost": 0, "symbols": 0}

    # -- link adaptation ---------------------------------------------------- #
    def select_mcs(self, table_path, bler_target):
        table_path = table_path or os.path.join(os.path.dirname(__file__), "mcs_table.json")
        if not os.path.isfile(table_path):
            raise FileNotFoundError(
                f"{table_path} missing - run python -m baselines.traditional.calibrate_mcs first, "
                f"or pass mcs=(bits_per_symbol, rate).")
        tab = json.load(open(table_path))
        key = "cdl" if self.channel == "cdl" else self.channel
        rows = tab["bler"][key]                      # {mcs_name: {snr: bler}}
        best, best_eff = None, -1.0
        for m in MCS_TABLE:
            curve = rows[mcs_name(m)]
            snrs = sorted(float(s) for s in curve)
            s = min(snrs, key=lambda x: abs(x - self.snr_db))      # nearest calibrated SNR
            if curve[f"{s:g}"] <= bler_target and m[0] * m[1] > best_eff:
                best, best_eff = m, m[0] * m[1]
        return best or MCS_TABLE[0]                  # nothing meets the target -> most robust

    # -- plumbing ------------------------------------------------------------ #
    def _codec(self, k: int, n: int, nbps: int):
        key = (k, n)
        if key not in self._enc:
            e = LDPC5GEncoder(k, n)
            self._enc[key] = e
            self._dec[key] = LDPC5GDecoder(e, hard_out=True, return_infobits=True, num_iter=self.num_iter)
        if nbps not in self._map:
            self._map[nbps] = Mapper("qam", num_bits_per_symbol=nbps)
            self._dem[nbps] = Demapper("app", "qam", num_bits_per_symbol=nbps)
        return self._enc[key], self._dec[key], self._map[nbps], self._dem[nbps]

    def _gains(self, m: int) -> torch.Tensor:
        if self.channel == "awgn":
            return torch.ones(m, dtype=torch.complex64, device=self.device)
        if self.channel == "rayleigh":
            return torch.complex(torch.randn(m, device=self.device), torch.randn(m, device=self.device)) / math.sqrt(2.0)
        return self._cdl(m).to(self.device)

    def _send_block(self, bits: np.ndarray, nbps: int, rate: float):
        k = bits.size
        n = int(math.floor(k / rate / nbps) * nbps)      # rate >= nominal (BG1 needs rate > 1/3)
        n = max(n, int(math.ceil((k + 1) / nbps) * nbps))
        enc, dec, mapper, demapper = self._codec(k, n, nbps)
        u = torch.from_numpy(bits.astype(np.float32))[None].to(self.device)
        x = mapper(enc(u)).reshape(-1)
        h = self._gains(x.numel()).to(x.dtype)
        no = 10.0 ** (-self.snr_db / 10.0)
        noise = math.sqrt(no / 2.0) * torch.complex(torch.randn(x.numel(), device=self.device),
                                                    torch.randn(x.numel(), device=self.device)).to(x.dtype)
        y = h * x + noise
        x_hat = (y / h).reshape(1, -1)
        no_eff = (no / h.abs().clamp(min=1e-12).pow(2)).reshape(1, -1)
        u_hat = dec(demapper(x_hat, no_eff))
        return u_hat.round().to(torch.uint8).cpu().numpy().reshape(-1), x.numel()

    # -- main ------------------------------------------------------------------ #
    @torch.no_grad()
    def transmit(self, payload: bytes):
        """Send one frame's bytes.  Returns (bytes | None, info)."""
        nbps, rate = self.mcs
        bits = bytes_to_bits(_crc_append(payload))
        nblk = max(1, math.ceil(bits.size / KMAX))
        per = math.ceil(bits.size / nblk)
        out, nsym = [], 0
        for b in range(nblk):
            chunk = bits[b * per:(b + 1) * per]
            rx, ns = self._send_block(chunk, nbps, rate)
            out.append(rx[:chunk.size]); nsym += ns
        got = _crc_check(bits_to_bytes(np.concatenate(out)))
        self.stats["frames"] += 1; self.stats["symbols"] += nsym; self.stats["lost"] += int(got is None)
        return got, {"n_symbols": nsym, "n_bits": int(bits.size), "mcs": mcs_name(self.mcs), "ok": got is not None}
