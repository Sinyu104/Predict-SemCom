"""
5G-NR-style uplink for the practical SSCC baseline (robot -> edge).

Per video frame (one transport block, TB):
  1. SNR -> MCS: highest-spectral-efficiency MCS from 3GPP TS 38.214 Table 5.1.3.1-1 whose TB error rate
     at that SNR is <= 10 % (the TS 38.214 CQI definition, bler_target=0.1).  BLER is calibrated by
     Monte Carlo at the actual TB size (calibrate()), per channel.
  2. MCS -> resource elements: the TB (payload + CRC32) is segmented into LDPC code blocks (TS 38.212 style,
     <= 8448 info bits each, rate-matched to Qm*R), needing N1 = bits / (Qm*R) REs per transmission.
  3. The robot has `re_budget` REs per frame period.  If N1 > re_budget the frame cannot be delivered in
     time -> OUTAGE (lost).  Otherwise it transmits and, on CRC failure, retransmits (HARQ, chase
     combining of LLRs) while the leftover budget allows, up to `max_tx` transmissions.
  4. CRC failure after the last transmission -> frame LOST (receiver gets nothing).

Channel: Sionna awgn / rayleigh / 3GPP CDL, perfect CSI, coherent ZF, N0 = 1/snr_lin (same as ours).
Simplifications (documented): MCS 0-2 omitted (Sionna LDPC has no repetition, rate < 1/5); TB CRC is 32 bit;
chase (not incremental-redundancy) HARQ; TB size = payload bits, not the quantized TS 38.214 TBS.
"""
from __future__ import annotations

import json, math, os, zlib
import numpy as np
import torch

from .bit_utils import bytes_to_bits, bits_to_bytes
from .sionna_channel import _CDLGains, _sn_config, LDPC5GEncoder, LDPC5GDecoder, Mapper, Demapper

KMAX = 8448


def nr_mcs_table():
    from sionna.phy.nr import decode_mcs_index
    out = []
    for i in range(3, 29):
        qm, r = decode_mcs_index(i, table_index=1, transform_precoding=False)[:2]
        out.append((i, int(qm), float(r)))
    return out


def _crc_append(b: bytes) -> bytes:
    return b + zlib.crc32(b).to_bytes(4, "big")


def _crc_ok(d: bytes):
    if len(d) < 4:
        return None
    return d[:-4] if zlib.crc32(d[:-4]).to_bytes(4, "big") == d[-4:] else None


class NRLink:
    def __init__(self, channel="awgn", snr_db=10.0, tb_bytes=1250, re_budget=8000, max_tx=4,
                 bler_target=0.1, cdl_model="C", table_path=None, device=None, seed=None, num_iter=20):
        self.channel, self.snr_db = channel, float(snr_db)
        self.tb_bytes, self.re_budget, self.max_tx = int(tb_bytes), int(re_budget), int(max_tx)
        self.num_iter = num_iter
        self.device = device or str(getattr(_sn_config, "device", "cpu"))
        if seed is not None:
            _sn_config.seed = seed; torch.manual_seed(seed)
        self._cdl = None
        if channel == "cdl":
            self._cdl = _CDLGains(device=self.device, model=cdl_model, delay_spread=100e-9,
                                  carrier_frequency=3.5e9, max_speed=3.0)
        self._enc, self._dec, self._map, self._dem = {}, {}, {}, {}
        self.table = nr_mcs_table()
        self.table_path = table_path or os.path.join(os.path.dirname(__file__), f"nr_bler_tb{self.tb_bytes}.json")
        self.mcs = self.select_mcs(bler_target)
        self.stats = dict(frames=0, lost=0, outage=0, symbols=0, tx=0)

    # -- plumbing ----------------------------------------------------------- #
    def _codec(self, k, n, qm):
        if (k, n) not in self._enc:
            e = LDPC5GEncoder(k, n); self._enc[(k, n)] = e
            self._dec[(k, n)] = LDPC5GDecoder(e, hard_out=True, return_infobits=True, num_iter=self.num_iter)
        if qm not in self._map:
            self._map[qm] = Mapper("qam", num_bits_per_symbol=qm)
            self._dem[qm] = Demapper("app", "qam", num_bits_per_symbol=qm)
        return self._enc[(k, n)], self._dec[(k, n)], self._map[qm], self._dem[qm]

    def _gains(self, m):
        if self.channel == "awgn":
            return torch.ones(m, dtype=torch.complex64, device=self.device)
        if self.channel == "rayleigh":
            return torch.complex(torch.randn(m, device=self.device), torch.randn(m, device=self.device)) / math.sqrt(2)
        return self._cdl(m).to(self.device)

    def _blocks(self, nbits, qm, r):
        kmax = 3824 if r <= 1 / 3 + 1e-6 else KMAX      # rate <= 1/3 => only BG2 (k <= 3824) in Sionna
        nblk = max(1, math.ceil(nbits / kmax)); per = math.ceil(nbits / nblk)
        sizes = [min(per, nbits - i * per) for i in range(nblk)]
        ns = [max(int(math.floor(k / r / qm) * qm), int(math.ceil((k + 1) / qm) * qm)) for k in sizes]
        return sizes, ns

    def _llr(self, x, qm):
        """Send symbols x [B, S] through the channel; return the demapper LLRs [B, S*qm]."""
        xs = x.reshape(-1)
        h = self._gains(xs.numel()).to(xs.dtype)
        no = 10.0 ** (-self.snr_db / 10.0)
        noise = math.sqrt(no / 2) * torch.complex(torch.randn(xs.numel(), device=self.device),
                                                  torch.randn(xs.numel(), device=self.device)).to(xs.dtype)
        y = h * xs + noise
        xh = (y / h).reshape(x.shape)
        ne = (no / h.abs().clamp(min=1e-12).pow(2)).reshape(x.shape)
        return self._dem[qm](xh, ne)

    # -- link adaptation / calibration --------------------------------------- #
    @torch.no_grad()
    def calibrate(self, snrs, blocks=150):
        """TB error rate per (MCS, SNR) for this channel at tb_bytes -> JSON."""
        nbits = (self.tb_bytes + 4) * 8
        tab = json.load(open(self.table_path)) if os.path.isfile(self.table_path) else {}
        cur = tab.setdefault(self.channel, {})
        old = self.snr_db
        for i, qm, r in self.table:
            sizes, ns = self._blocks(nbits, qm, r)
            row = cur.setdefault(str(i), {})
            for s in snrs:
                self.snr_db = float(s); fail = torch.zeros(blocks, dtype=torch.bool, device=self.device)
                for k, n in zip(sizes, ns):
                    enc, dec, mp, _ = self._codec(k, n, qm)
                    u = torch.randint(0, 2, (blocks, k), device=self.device).float()
                    llr = self._llr(mp(enc(u)), qm)
                    fail |= (dec(llr).round() != u).any(1)
                row[f"{s:g}"] = float(fail.float().mean())
        self.snr_db = old
        json.dump(tab, open(self.table_path, "w"), indent=1)

    def select_mcs(self, bler_target):
        if not os.path.isfile(self.table_path):
            raise FileNotFoundError(f"{self.table_path} missing - run: python -m baselines.traditional.calibrate_nr "
                                    f"--tb_bytes {self.tb_bytes}")
        tab = json.load(open(self.table_path)).get(self.channel)
        if not tab:
            raise KeyError(f"channel '{self.channel}' not calibrated in {self.table_path}")
        best, eff = None, -1.0
        for i, qm, r in self.table:
            row = tab.get(str(i))
            if not row:
                continue
            s = min((float(x) for x in row), key=lambda x: abs(x - self.snr_db))
            if row[f"{s:g}"] <= bler_target and qm * r > eff:
                best, eff = (i, qm, r), qm * r
        return best or self.table[0]

    # -- one frame --------------------------------------------------------------- #
    @torch.no_grad()
    def transmit(self, payload: bytes):
        _, qm, r = self.mcs
        bits = bytes_to_bits(_crc_append(payload))
        sizes, ns = self._blocks(bits.size, qm, r)
        n1 = sum(n // qm for n in ns)                               # REs per transmission
        self.stats["frames"] += 1
        if n1 > self.re_budget:                                      # cannot fit this frame period
            self.stats["lost"] += 1; self.stats["outage"] += 1; self.stats["symbols"] += self.re_budget
            return None, dict(ok=False, outage=True, n_symbols=self.re_budget, n_tx=0, mcs=self.mcs[0], re_per_tx=n1)
        chunks, p = [], 0
        for k in sizes:
            chunks.append(bits[p:p + k]); p += k
        cws = []
        for k, n, c in zip(sizes, ns, chunks):
            enc, dec, mp, _ = self._codec(k, n, qm)
            u = torch.from_numpy(c.astype(np.float32))[None].to(self.device)
            cws.append((enc, dec, mp, mp(enc(u)), k, n))
        llr_sum = [None] * len(cws); used = ntx = 0; got = None
        while ntx < self.max_tx and used + n1 <= self.re_budget:
            for j, (enc, dec, mp, x, k, n) in enumerate(cws):
                l = self._llr(x, qm)
                llr_sum[j] = l if llr_sum[j] is None else llr_sum[j] + l      # chase combining
            used += n1; ntx += 1
            out = np.concatenate([cws[j][1](llr_sum[j]).round().to(torch.uint8).cpu().numpy().reshape(-1)[:sizes[j]]
                                  for j in range(len(cws))])
            got = _crc_ok(bits_to_bytes(out))
            if got is not None:
                break
        self.stats["symbols"] += used; self.stats["tx"] += ntx; self.stats["lost"] += int(got is None)
        return got, dict(ok=got is not None, outage=False, n_symbols=used, n_tx=ntx, mcs=self.mcs[0], re_per_tx=n1)
