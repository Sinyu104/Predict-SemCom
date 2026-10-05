"""Calibrate BLER per (channel, MCS, SNR) for link adaptation -> mcs_table.json.

BLER is measured at a representative information length (default k=1500 bits, ~ a low-QP P-frame)
with Monte-Carlo blocks.  AdaptiveLink picks, per SNR, the highest-throughput MCS with BLER <= target.

    python -m baselines.traditional.calibrate_mcs --channels awgn rayleigh cdl --blocks 400
"""
import argparse, json, math, os
import numpy as np, torch
from .link import AdaptiveLink, MCS_TABLE, mcs_name


@torch.no_grad()
def bler(link: AdaptiveLink, k: int, nbps: int, rate: float, blocks: int) -> float:
    n = int(math.floor(k / rate / nbps) * nbps)
    enc, dec, mapper, demapper = link._codec(k, n, nbps)
    u = torch.randint(0, 2, (blocks, k), device=link.device).float()
    x = mapper(enc(u))                                     # [B, n/nbps]
    xs = x.reshape(-1)
    h = link._gains(xs.numel()).to(xs.dtype)
    no = 10.0 ** (-link.snr_db / 10.0)
    noise = math.sqrt(no / 2.0) * torch.complex(torch.randn(xs.numel(), device=link.device),
                                                torch.randn(xs.numel(), device=link.device)).to(xs.dtype)
    y = h * xs + noise
    x_hat = (y / h).reshape(blocks, -1)
    no_eff = (no / h.abs().clamp(min=1e-12).pow(2)).reshape(blocks, -1)
    u_hat = dec(demapper(x_hat, no_eff)).round()
    return float((u_hat != u).any(dim=1).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=["awgn", "rayleigh", "cdl"])
    ap.add_argument("--cdl_model", default="C")
    ap.add_argument("--snrs", type=float, nargs="+", default=list(np.arange(-2, 25, 1.0)))
    ap.add_argument("--k", type=int, default=1500)
    ap.add_argument("--blocks", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "mcs_table.json"))
    a = ap.parse_args()
    res = {"k": a.k, "blocks": a.blocks, "cdl_model": a.cdl_model, "bler": {}}
    for ch in a.channels:
        link = AdaptiveLink(ch, 0.0, mcs=MCS_TABLE[0], cdl_model=a.cdl_model, seed=0)
        res["bler"][ch] = {}
        for m in MCS_TABLE:
            curve = {}
            for s in a.snrs:
                link.snr_db = float(s)
                curve[f"{s:g}"] = bler(link, a.k, m[0], m[1], a.blocks)
            res["bler"][ch][mcs_name(m)] = curve
            print(f"{ch:8} {mcs_name(m):14} BLER<=1e-2 from SNR >= "
                  f"{next((s for s in a.snrs if curve[f'{s:g}'] <= 1e-2), None)} dB", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
