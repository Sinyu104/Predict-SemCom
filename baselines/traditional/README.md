# Traditional SSCC Baseline (H.264/H.265 + LDPC + Sionna)

Separate source–channel coding baseline — the classical foil to the JSCC
semantic communication system. It transmits robot camera frames over a
5G-LDPC coded link and an **NVIDIA Sionna** channel (flat Rayleigh or 3GPP
CDL), and is meant to expose the **cliff effect**: above a threshold SNR the
LDPC code corrects every bit and reconstruction is near-lossless; below it the
video bitstream is corrupted and the decoder collapses.

## Pipeline

```
frames ─▶ H.264/H.265 ─▶ bits ─▶ 5G-LDPC encode ─▶ QAM map
       ─▶ [ per-symbol fading h + AWGN ]   (NVIDIA Sionna)
       ─▶ ZF equalize (x̂=y/h) ─▶ QAM demap (LLR) ─▶ 5G-LDPC decode ─▶ bits
       ─▶ H.264/H.265 decode ─▶ reconstructed frames
```

All three channels share one coherent zero-forcing receiver (per-symbol
effective noise `N0/|h|²`); only the per-symbol complex gain `h` differs:

| `--channel` | gain `h`                                   | use                                   |
| ----------- | ------------------------------------------ | ------------------------------------- |
| `awgn`      | `1`                                        | rate-distortion reference (no fading) |
| `rayleigh`  | `CN(0,1)` i.i.d. per symbol (flat)         | matches `models.RayleighChannel`      |
| `cdl`       | Sionna TR-38.901 CDL, OFDM freq. response  | realistic 3GPP frequency/time-selective |
| `none`      | perfect bit pipe (no Sionna)               | test codec/metrics without Sionna     |

Constellations are unit-energy and channels are normalized (`E|h|²=1`), so
`N0 = 1/snr_lin` for every channel — the SNR axis is comparable across all of
them and consistent with `models.RayleighChannel`.

## Environment

Sionna **2.0.1** is installed in the **`worldmodel` conda env** and is
**PyTorch-based** (no TensorFlow). ffmpeg (libx264/libx265) + PyAV provide the
codec. Run everything with that env:

```bash
conda activate worldmodel
```

Sionna 2.0 defaults to **CUDA**; pass `--device cpu` to keep the baseline off a
GPU that training is using. The codec/metrics path (`--channel none`) needs
neither Sionna nor a GPU.

## Rate control (rate-distortion)

Pick one knob in `run_baseline.py`:

- `--qp Q [Q ...]` — **constant QP** (fixed quantizer). Clean, monotonic
  rate-vs-quality points; sweep a list to trace a rate-distortion curve.
  Range 0 (lossless) .. 51 (worst).
- `--crf N` — constant-rate-factor (perceptual VBR), used when `--qp` is unset.

Measured RD (H.265, 14-frame clip, perfect channel):

| QP | bpp    | PSNR    |
| -- | ------ | ------- |
| 18 | 0.186  | 40.1 dB |
| 28 | 0.104  | 37.9 dB |
| 38 | 0.073  | 34.0 dB |

## Usage

```bash
conda activate worldmodel

# Rate-distortion sweep over QP at a fixed 12 dB CDL-C channel:
python -m baselines.traditional.run_baseline \
    --input data/franka_demos.hdf5 --episode episode_0 --num_frames 14 \
    --codec h265 --qp 18 24 30 36 42 --channel cdl --cdl_model C --snr_db 12 \
    --csv baselines/traditional/rd_cdlC.csv

# SNR sweep (cliff effect) at fixed quality, Rayleigh:
python -m baselines.traditional.run_baseline \
    --input data/franka_demos.hdf5 --qp 28 --channel rayleigh \
    --snr_db -4 -2 0 2 4 6 8 --csv baselines/traditional/snr_rayleigh.csv

# Codec + metrics only, WITHOUT Sionna:
python -m baselines.traditional.run_baseline \
    --input data/franka_demos.hdf5 --qp 28 --channel none

# Plot from CSVs:
python -m baselines.traditional.plot_curves --csv baselines/traditional/rd_cdlC.csv \
    --mode rd --out baselines/traditional/rd.png
python -m baselines.traditional.plot_curves --csv baselines/traditional/snr_rayleigh.csv \
    --mode snr --out baselines/traditional/snr.png
```

### Key flags

| Flag                       | Meaning                                         | Default   |
| -------------------------- | ----------------------------------------------- | --------- |
| `--codec`                  | `h264` / `h265`                                 | `h265`    |
| `--qp` / `--crf`           | rate control (QP list sweeps RD; CRF otherwise) | crf 28    |
| `--channel`                | `awgn` / `rayleigh` / `cdl` / `none`            | `rayleigh`|
| `--snr_db`                 | one or more SNRs to sweep                        | `10`      |
| `--k` / `--n` / `--bps`    | LDPC info/codeword length, QAM bits/symbol       | 3072/6144/2 |
| `--cdl_model`              | CDL profile A–E                                  | `C`       |
| `--delay_spread`           | RMS delay spread (s)                             | `100e-9`  |
| `--carrier_freq`           | carrier frequency (Hz)                           | `3.5e9`   |
| `--speed`                  | max UE speed (m/s) → Doppler                     | `3.0`     |
| `--device`                 | `cuda` / `cpu`                                    | Sionna's  |
| `--csv` / `--out`          | results table / montage PNG dir                  | —         |

Each row reports source **bpp**, code rate, **post-LDPC BER**, frames-decoded,
mean **PSNR**, and OK/FAIL (FAIL = channel corrupted the bitstream enough to
break the decoder).

## Metrics — what to report and why

All bit-level metrics are measured **after LDPC decoding** (on the recovered
information bits), not on the raw channel.

| Metric        | Definition                                             | What it tells you                              |
| ------------- | ------------------------------------------------------ | ---------------------------------------------- |
| **BER**       | wrong bits ÷ total bits                                | channel-level quality; **understates** image damage because errors are bursty and location-dependent |
| **BLER**      | codewords with ≥1 wrong bit ÷ total codewords (k=3072) | "did corruption enter the bitstream" — closer to whether a frame survives |
| **frames intact** | frames bit-identical to the clean (no-channel) decode | the **honest, image-domain** metric; this is what to plot against JSCC |

Why not just BER: a low BER can still destroy the video. H.265 uses stateful
entropy coding (CABAC) + inter-frame prediction, so one uncorrected error
corrupts from its position to the next resync point and **propagates across the
GOP** to the next keyframe. Measured on episode_0: **Rayleigh @ 3 dB has BER ≈
1.1 % yet ~0 of 300 frames survive.** Report **frames-intact / PSNR-vs-original**
as the headline; use BER/BLER for the coding-theory view.

**Statistics.** A single channel realization is *not* enough near the cliff —
the transition region and low-BER tail are dominated by rare block errors and
have high variance. Use **many realizations per SNR point** and report a
confidence interval (see `Validated behaviour`). One point = one SNR value;
"200 realizations" = 200 independent channel draws at that SNR, averaged.

## Files

| File                | Role                                                          |
| ------------------- | ------------------------------------------------------------- |
| `video_codec.py`    | H.264/H.265 encode/decode (PyAV); QP/CRF; corruption-robust   |
| `sionna_channel.py` | LDPC + QAM + Sionna channel (awgn/rayleigh/cdl); `transmit()` |
| `bit_utils.py`      | byte ↔ bit packing (no Sionna/torch dependency)              |
| `run_baseline.py`   | end-to-end CLI + QP×SNR sweep + CSV                          |
| `plot_curves.py`    | RD (PSNR-vs-bpp) and SNR (PSNR-vs-SNR) plots from CSVs        |

## Validated behaviour

All runs on `data/franka_demos.hdf5`, H.265 QP 28, QPSK, rate-½ LDPC (k=3072,
n=6144), Sionna 2.0 on CUDA.

**QP rate-distortion** (14-frame clip, perfect channel) — clean monotonic curve:
QP 18 → 0.186 bpp / 40.1 dB, QP 28 → 0.104 / 37.9, QP 38 → 0.073 / 34.0.

**Monte-Carlo BER / BLER** (200 realizations per SNR point ≈ 14.5 M bits;
mean ± 95 % CI). All channels show a sharp waterfall in a ~1–3 dB window:

| SNR  | AWGN BER | Rayleigh BER | CDL-A BER | CDL-C BER |
| ---- | -------- | ------------ | --------- | --------- |
| 0 dB | 1.6e-1   | 2.5e-1       | 1.8e-1    | 2.2e-1    |
| 1 dB | 8.3e-3   | 2.1e-1       | 5.6e-2    | 1.6e-1    |
| 2 dB | 0        | 1.6e-1       | 5.0e-3    | 4.7e-2    |
| 3 dB | 0        | 1.1e-2       | 4.8e-4    | 6.4e-3    |
| 4 dB | 0        | 0            | 2.7e-5    | 7.5e-4    |
| 5 dB | 0        | 0            | 0         | 1.6e-4    |
| 6 dB | 0        | 0            | 0         | 0         |

Waterfall midpoint (BLER ≈ 0.5), best → worst: **AWGN (~1 dB) < CDL-A (~1.3 dB)
< CDL-C (~2 dB) ≈ Rayleigh (~3 dB)** — flat Rayleigh is worst (no frequency
diversity); CDL's multipath gives the code frequency diversity. `BER = 0` means
"0 errors observed in ~14.5 M bits" (< ~1e-7 here), not a proven floor.

**Frames intact — image-domain cliff** (full episode_0, 300 frames, GOP 12,
20 realizations/point; a frame counts if bit-identical to the clean decode):

| SNR  | Rayleigh (mean ± std)  | CDL-C (mean ± std)          |
| ---- | ---------------------- | --------------------------- |
| 2 dB | 0 ± 0                  | 0 ± 0                       |
| 3 dB | 0.3 ± 1.3 (max 6)      | 27.6 ± 32.8 (max 140)       |
| 4 dB | **300 ± 0**            | 171.6 ± 89 (min 35, max 300)|
| 5 dB | 300 ± 0               | 275.9 ± 67.8 (min 67)       |
| 6 dB | 300 ± 0               | **300 ± 0**                 |

All-or-nothing: Rayleigh flips 0 → 300 frames between 3 and 4 dB. CDL-C
degrades over 3–6 dB with **large variance** (one fading draw can spare or wipe
long stretches — the GOP-propagation effect), which is exactly why single-run
numbers mislead and many realizations are needed.

## Notes / caveats

- **Genie frame length.** The receiver is told the compressed byte length (no
  header sent). A deployed system would protect a short length header
  separately; fine for a rate–distortion baseline.
- **Robustness on purpose.** `decode_frames` catches decoder exceptions, drops
  frames the corrupted stream resyncs to at the wrong geometry, and pads missing
  frames with gray — the FAIL/cliff behaviour is the *result*, not a bug.
- **CDL as a symbol-stream channel.** CDL is evaluated as an OFDM frequency
  response and its per-subcarrier gains are laid over the QAM symbol stream
  (frequency- and time-selective fading). This is a link-level abstraction, not
  a full OFDM waveform simulation.
- **Comparison to JSCC.** Sweep `--snr_db` here and against the semantic
  pipeline; SSCC sits near the codec ceiling until it falls off a cliff, while
  JSCC should degrade gracefully.
