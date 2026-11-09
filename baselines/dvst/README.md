# DVST baseline (Deep Video Semantic Transmission)

Wang et al., *Wireless Deep Video Semantic Transmission*, IEEE JSAC 41(1), 2023 — the learned
temporal-context JSCC foil to our Wyner-Ziv codec. It exploits the **previous reconstructed
frame** as context (conditional coding + entropy-model-driven rate allocation), whereas ours
uses a **world-model prediction** `ẑ_t` conditioned on the action. Same source (VAE latent
`z_t`, 4×28×28), same channels (`awgn`/`rayleigh`/`cdl`, coherent ZF, `N0 = P/snr`), same cache and
held-out split as `experiments/stage2_twophase/train_masked_alloc_wz.py`.

| Paper                                  | Code (`model.py`)                         |
| -------------------------------------- | ----------------------------------------- |
| contextual analysis `g_a(x\|c̃)`        | `DVST.g_a`                                |
| contextual deep JSCC `f_e / f_d`       | `DVST.fe_*` / `fd_*` (+ rate tokens)      |
| synthesis `g_s(ŷ\|ĉ)`                  | `DVST.g_s`                                |
| contexts `c̃` (feature), `d̃` (codeword) | `ContextGen` (`phi`, `gam`)               |
| entropy model (hyper + AR + temporal)  | `EntropyModel` (Laplace, Eq. 7–9)         |
| `k_{t,i} = η·(−log₂P)`, set 𝒱 (Eq. 10) | `DVST.allocate` (STE), `levels`           |
| progressive training (Eq. 20–24)       | `train.py` phase 1 → phase 2              |

The transmitter sees only `z_t` and the previous frame (WZ-clean); the receiver needs no entropy
decoding, only the quantised per-cell allocation `k_{t,i}` (side info, ≈ log₂|𝒱| bits per cell).

## Run (predictor env, pick a free GPU)

```bash
CUDA_VISIBLE_DEVICES=2 python baselines/dvst/train.py --pre outputs/rollout_disturbed_clean_s2_moe \
    --out outputs/dvst_lam0.1 --lam 0.1 --snr_min 0 --snr_max 20 --channel awgn   # sweep --lam for the RD curve
CUDA_VISIBLE_DEVICES=2 python baselines/dvst/eval.py --ckpt outputs/dvst_lam0.1/dvst.pt \
    --pre outputs/rollout_disturbed_clean_s2_moe --snr 0 5 10 15 20                # -> results/dvst.csv
```

`eval.py` appends stamped rows (repo CSV convention, `beta` = DVST λ, `k` = mean sent reals/cell).
`frac16` = rate as a fraction of the WZ codec's full 16-channel rate, so DVST and WZ share a rate axis.

## Deviations from the paper (adaptations to latent space)

- **No motion link / warping**: context comes from the previous frame directly (zero motion);
  optical flow on a 28×28 latent grid is unreliable.
- **Conv residual backbones instead of Swin**; one embedding = one latent cell (not a 32×32 px patch).
- **Hyperprior `z`** uses a per-channel factorised logistic density (not Ballé's CDF MLP).
- **Reference frame**: previous latent passed through the codec *without channel/context* (Eq. 2,
  intra mode; trained with `--p_intra`), mixed with the lossless previous latent (`--ref mix`,
  paper step 3). There is no GOP unrolling because the cache holds single `(z_t, z_{t-1})` pairs; at
  deploy the receiver should feed back its own `ẑ_{t-1}` / `ŷ_{t-1}` (not implemented in `eval.py`,
  which uses the intra reference — an optimistic reference, noted when reporting).
- Distortion is plain full-frame latent MSE (no disturbance weighting — a baseline shouldn't know
  about the disturbed object). λ here multiplies CBR (reals per source real), not raw k.

## Online (closed loop, PI0Fast, two cameras)

`server/semcom_server.py --mode dvst --dvst_ckpt <ckpt> --snr_db 10 --channel awgn` (policy server and ROS
bridge exactly as for `wz2cam`; no Ctrl-World / Stage-2 ckpts needed). Per request: both cameras are
VAE-encoded into one 4x28x56 latent; the transmitter encodes with its noise-free simulated reference, the
receiver decodes with its actually-received reference (Fig. 2); the first request of an episode is intra.
Each request logs `frac16` and latent MSE. **Train on the matching data**: online frames are one per
30-step chunk and two-camera, so use `--pre outputs/rollout_disturbed_2cam outputs/rollout_calm_2cam`
(ZT/ZPREV 30 steps apart, 28x56), not the single-camera cache.

## Not done yet

Latent MSE is not the metric that matters here (see CLAUDE.md). To score with the VLA, decode
`z_hat` with the VAE and feed `score_arms*.py`, as `decode_arms_alloc.py` does for the WZ arms.
Trained results are not included; the smoke run (1+1 epochs) only verified the pipeline.
