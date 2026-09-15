"""
End-to-end traditional SSCC baseline:
    HDF5 frames -> H.264/H.265 -> LDPC+QAM -> Sionna channel -> decode -> metrics

Channels (Sionna 2.0): awgn | rayleigh (flat) | cdl (3GPP TR-38.901).
Rate control: constant QP (for clean rate-distortion curves) or CRF.

Examples
--------
# Rate-distortion sweep over QP at a fixed 12 dB CDL-C channel, H.265:
python -m baselines.traditional.run_baseline \
    --input data/franka_demos.hdf5 --episode episode_0 --num_frames 14 \
    --codec h265 --qp 18 24 30 36 42 --channel cdl --cdl_model C --snr_db 12 \
    --csv baselines/traditional/rd_cdlC.csv

# SNR sweep (cliff effect) at fixed quality, Rayleigh vs CDL:
python -m baselines.traditional.run_baseline \
    --input data/franka_demos.hdf5 --qp 28 --channel rayleigh --snr_db 0 3 6 9 12 15

# Codec + metrics only, WITHOUT Sionna (perfect bit pipe):
python -m baselines.traditional.run_baseline \
    --input data/franka_demos.hdf5 --qp 28 --channel none
"""

from __future__ import annotations

import argparse
import csv as _csv
import os

import h5py
import numpy as np

from .video_codec import encode_frames, decode_frames
from .bit_utils import bytes_to_bits, bits_to_bytes


# ── data loading ────────────────────────────────────────────────────────── #
_OBS_KEYS = ["observations", "observations_cam1", "observations_cam2"]


def _resolve_obs_key(grp, preferred: str | None):
    for k in ([preferred] if preferred else []) + _OBS_KEYS:
        if k and k in grp:
            return k
    return None


def load_frames(path, episode, obs_key, start, num_frames) -> np.ndarray:
    """Return (T, H, W, 3) uint8 RGB frames from an HDF5 demo file."""
    with h5py.File(path, "r") as f:
        flat = _resolve_obs_key(f, obs_key)
        if flat is not None:                       # layout A: flat file
            ds = f[flat]
        else:                                      # layout B: episode groups
            ep = episode or sorted(f.keys())[0]
            grp = f[ep]
            key = _resolve_obs_key(grp, obs_key)
            if key is None:
                raise KeyError(f"No observations dataset in group '{ep}'.")
            ds = grp[key]
        T = len(ds)
        end = T if num_frames < 0 else min(start + num_frames, T)
        frames = np.array(ds[start:end])
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


# ── metrics ─────────────────────────────────────────────────────────────── #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse <= 1e-8 else float(10.0 * np.log10(255.0 ** 2 / mse))


def save_montage(orig, recon, path, n: int = 4):
    """Save a top=original / bottom=reconstruction strip for up to n frames."""
    try:
        from PIL import Image
    except ImportError:
        print("[viz] Pillow not installed — skipping montage.")
        return
    T = orig.shape[0]
    idx = np.linspace(0, T - 1, min(n, T)).round().astype(int)
    top = np.concatenate([orig[i]  for i in idx], axis=1)
    bot = np.concatenate([recon[i] for i in idx], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(grid).save(path)


# ── one (quality, snr) point ────────────────────────────────────────────── #
def run_once(frames, args, snr_db, qp, ch_cache) -> dict:
    T, H, W, _ = frames.shape

    # 1) Source coding (QP overrides CRF when set).
    bitstream, stream_fmt = encode_frames(
        frames, codec=args.codec, crf=args.crf, qp=qp, gop=args.gop
    )
    n_src_bits = len(bitstream) * 8
    bpp = n_src_bits / (T * H * W)

    # 2) Channel coding + transmission.
    if args.channel == "none":                     # perfect bit pipe (codec test)
        rx_bytes, ber, rate = bitstream, 0.0, 1.0
    else:
        key = (args.channel, snr_db)
        ch = ch_cache.get(key)
        if ch is None:
            from .sionna_channel import SionnaCodedChannel
            ch = SionnaCodedChannel(
                k=args.k, n=args.n, num_bits_per_symbol=args.bps,
                snr_db=snr_db, channel=args.channel, num_iter=args.ldpc_iter,
                device=args.device, seed=args.seed,
                model=args.cdl_model, delay_spread=args.delay_spread,
                carrier_frequency=args.carrier_freq, max_speed=args.speed,
            ) if args.channel == "cdl" else SionnaCodedChannel(
                k=args.k, n=args.n, num_bits_per_symbol=args.bps,
                snr_db=snr_db, channel=args.channel, num_iter=args.ldpc_iter,
                device=args.device, seed=args.seed,
            )
            ch_cache[key] = ch
        rx_bits, st = ch.transmit(bytes_to_bits(bitstream))
        rx_bytes, ber, rate = bits_to_bytes(rx_bits), st["ber"], st["code_rate"]

    # 3) Source decoding (robust to corruption).
    recon, n_decoded = decode_frames(rx_bytes, stream_fmt, T, H, W)

    # 4) Metrics.
    per_frame = [psnr(frames[i], recon[i]) for i in range(n_decoded)]
    mean_psnr = float(np.mean(per_frame)) if per_frame else 0.0
    res = {
        "channel": args.channel, "snr_db": snr_db, "codec": args.codec,
        "qp": qp if qp is not None else "", "crf": "" if qp is not None else args.crf,
        "bpp": round(bpp, 5), "src_bits": n_src_bits, "code_rate": rate,
        "post_ldpc_ber": ber, "frames": T, "frames_decoded": n_decoded,
        "psnr_db": round(mean_psnr, 3), "decode_ok": int(n_decoded == T),
    }
    if args.out:
        q = f"qp{qp}" if qp is not None else f"crf{args.crf}"
        tag = f"{args.channel}_snr{snr_db:g}_{q}"
        save_montage(frames, recon, os.path.join(args.out, f"{tag}.png"))
    return res


def _fmt(r: dict) -> str:
    q = f"qp={r['qp']}" if r["qp"] != "" else f"crf={r['crf']}"
    return (f"{r['channel']:>8} SNR={float(r['snr_db']):>5.1f}dB {q:>7}  "
            f"bpp={r['bpp']:.4f} rate={r['code_rate']:.2f} "
            f"BER={r['post_ldpc_ber']:.2e} dec={r['frames_decoded']}/{r['frames']} "
            f"PSNR={r['psnr_db']:6.2f}dB {'OK' if r['decode_ok'] else 'FAIL'}")


def main():
    p = argparse.ArgumentParser(description="Traditional SSCC (H.26x + LDPC + Sionna) baseline.")
    # data
    p.add_argument("--input", required=True, help="HDF5 demo file.")
    p.add_argument("--episode", default=None, help="Episode group (layout B). Default: first.")
    p.add_argument("--obs_key", default=None, help="observations / _cam1 / _cam2.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=14, help="-1 = all.")
    # source coding
    p.add_argument("--codec", default="h265", choices=["h264", "h265"])
    p.add_argument("--crf", type=int, default=28, help="CRF quality (used if --qp unset).")
    p.add_argument("--qp", type=int, nargs="+", default=None,
                   help="Constant QP value(s); sweep for a rate-distortion curve.")
    p.add_argument("--gop", type=int, default=12)
    # channel
    p.add_argument("--channel", default="rayleigh",
                   choices=["awgn", "rayleigh", "cdl", "none"])
    p.add_argument("--snr_db", type=float, nargs="+", default=[10.0],
                   help="One or more SNRs (dB) to sweep.")
    p.add_argument("--k", type=int, default=3072, help="LDPC info block length.")
    p.add_argument("--n", type=int, default=6144, help="LDPC codeword length.")
    p.add_argument("--bps", type=int, default=2, help="QAM bits/symbol (2/4/6).")
    p.add_argument("--ldpc_iter", type=int, default=20)
    p.add_argument("--device", default=None, help="torch device (cuda/cpu). Default: Sionna's.")
    p.add_argument("--seed", type=int, default=0)
    # CDL (3GPP TR-38.901)
    p.add_argument("--cdl_model", default="C", choices=["A", "B", "C", "D", "E"])
    p.add_argument("--delay_spread", type=float, default=100e-9, help="RMS delay spread (s).")
    p.add_argument("--carrier_freq", type=float, default=3.5e9, help="Carrier frequency (Hz).")
    p.add_argument("--speed", type=float, default=3.0, help="Max UE speed (m/s) -> Doppler.")
    # output
    p.add_argument("--out", default=None, help="Dir to save montage PNGs.")
    p.add_argument("--csv", default=None, help="Write results table to this CSV.")
    args = p.parse_args()

    frames = load_frames(args.input, args.episode, args.obs_key,
                         args.start, args.num_frames)
    print(f"[data] {frames.shape[0]} frames @ {frames.shape[1]}x{frames.shape[2]} "
          f"from {args.input}")
    if args.channel == "cdl":
        print(f"[cdl] model={args.cdl_model} DS={args.delay_spread*1e9:g}ns "
              f"fc={args.carrier_freq/1e9:g}GHz vmax={args.speed}m/s")

    qp_list = args.qp if args.qp is not None else [None]     # None -> use CRF
    ch_cache: dict = {}
    rows: list[dict] = []
    print("=" * 92)
    for qp in qp_list:
        for snr in args.snr_db:
            r = run_once(frames, args, snr, qp, ch_cache)
            rows.append(r)
            print(_fmt(r))
    print("=" * 92)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
