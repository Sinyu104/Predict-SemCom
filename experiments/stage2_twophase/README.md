# Stage 2 — two-phase training

Scripts that produced the first working Wyner-Ziv channel in this project. Kept here
because the result is not otherwise reproducible, and because the ruled-out table below
is worth more than the code.

## The problem they solved

Seven Stage-2 architectures in a row had the transmitted signal contributing **~0.00%**
to reconstruction. Deleting the transmitter changed the output by less than a tenth of a
percent, at every disturbance level, on every frame.

The cause, measured rather than guessed:

```
|dL/ds~| relative, 256 most-disturbed frames
  randomly initialised   1.724e-03
  trained                3.268e-06     <- 500x collapse
```

`dL/dtheta_encoder` reaches the encoder **only** through `dL/ds~`. The decoder holds
`z_hat` (~95% correct), so suppressing a noisy `s~` early is locally optimal — and doing
so zeroes the gradient that would have taught the encoder what to encode. Backprop
worked; it found a good local optimum that ignores the channel. Encoder and decoder
cannot escape it alone.

## The recipe

```
Phase 1 (3 epochs)  z_hat WITHHELD from the refinement entirely — input, conditioning
                    AND residual base. The decoder cannot emit anything without decoding
                    s~, so it is forced to build the pathway.
Phase 2 (5 epochs)  z_hat returns, 30% z_hat-dropout with NO residual base when dropped.
                    Teaches WHEN to correct.
```

**Both phases use the full distribution.** Filtering to high-deviation frames was tested
in both phases and is worse in both:

| phase 1 | phase 2 | A vs B | A vs C | note |
|---|---|---|---|---|
| all data | all data | **+59.9%** | **+26.3%** | best |
| top 20% | all data | +24.9% | +19.2% | filtering phase 1 halves the channel |
| top 20% | top 20% | +5.3% | -8.3% | filtering phase 2 wrecks calm frames (D1 -104.8%) |

Wyner-Ziv is preserved: the encoder never sees `z_hat` in either phase. `z_hat` enters
only through the SideInfoEncoder prior, i.e. the conditional rate term.

## Result (full distribution, 70,368 samples)

```
A (with s~) 0.03071    B (s~ = 0) 0.07662    C (z_hat alone) 0.04166

A vs B = +59.92%    channel contribution
A vs C = +26.28%    system vs raw prediction
70,352 / 70,368 frames helped by >1%   (100.0%)

decile   A vs B    A vs C
D1      +74.97%     +0.9%     (calmest)
D5      +66.04%    +20.1%
D10     +43.45%    +37.3%     (most disturbed)
```

Positive in all ten deciles. `|dL/ds~|` climbed through phase 1 (1.0e-04 -> 2.7e-04) and
held through phase 2 (1.8e-04 -> 2.4e-04), versus 3.27e-06 when it collapses.

## Ruled out — do not re-test

| tried | result |
|---|---|
| encoder capacity (mean-pool -> flatten) | necessary, not sufficient |
| train/inference mismatch (noise `z_hat`, not `z_t`) | necessary, not sufficient |
| SDEdit noise level | t''=0 optimal; error rises monotonically with t'' |
| dedicated `s~` cross-attention + 16 tokens | no effect alone |
| deviation-weighted distortion | 2.65x gradient shift, no effect alone |
| **beta = 0** | **250 bits/frame transmitted, still +0.01%** |
| `z_hat`-dropout keeping the residual base | ineffective — the net emits delta=0 and coasts |
| filtering training data by deviation | worse in both phases, see table above |

beta is not the lever: at beta=0 the encoder sends 250 bits/frame and the decoder still
ignores them. That changes the encoder's *incentive*, not the decoder's *necessity*.

## Usage

The schedule is now in `Stage2Trainer` (`config: stage2_phase1_epochs`), so the normal
path is:

```bash
python -m torch.distributed.run --nproc_per_node=4 main.py --train --stage 2 --disturbed \
    --tasks pick_red_cube_to_tray pick_blue_cube_to_tray ... \
    --stage1_ckpt outputs/stage1_5cube_cam1_K8/stage1_best.pt \
    --output_data_dir outputs/stage2 --epoch 8
```

The scripts here are the fast path — they run off a cached `z_t`/`z_hat` so epochs take
minutes instead of hours, which is what made the ablations above affordable:

```bash
# 1. cache z_t and z_hat (one process per GPU, ~1.6 h; needs the Stage-1 checkpoint)
for i in 0 1 2 3; do
  python precompute.py --shard $i --nshards 4 --stride 3 --out pre/shard$i.pt &
done; wait

# 2. two-phase training
python twophase.py --pre pre --out ../../outputs/stage2_twophase \
    --top 1.0 --top2 1.0 --ep1 3 --ep2 5 --drop2 0.30 --bs 16

# 3. evaluation
python eval_full.py --pre pre --ckpt ../../outputs/stage2_twophase/phase2.pt
python tiny_tdp.py                                  # A/B/C by deviation group and t''
python persample.py --cache pre/zcache.pt           # per-sample benefit distribution
python gradcheck.py --cache pre/zcache.pt           # |dL/ds~| trained vs random init
```

Batch size 16 — the 784-token attention needs ~1.26 GB per layer at batch 64.

The cache is derived data and is NOT kept. It is tied to a specific Stage-1 checkpoint
with no provenance record, so retraining Stage 1 silently invalidates it; regenerate with
step 1 rather than reusing an old one.

## Measurement traps hit during this investigation

- Group disturbed frames at the **top 5%** (`pose_prob=0.05` in the collector) when
  *analysing*. Using the top 20% diluted the group ~4x and hid the effect. (This is about
  analysis grouping — do not confuse it with training-set filtering, which is harmful.)
- Training-log `rate` shows a spiky tail from reparameterisation noise and optimiser
  transients. It is **not** evidence of rate adaptation. Spikes landing in *adjacent*
  batches under a shuffled sampler proves they are optimiser artifacts.
- Always check the **per-sample** benefit distribution, not just group means. A mean over
  62 samples cannot show an effect concentrated in a handful of frames.
- Always check the **per-decile** breakdown, not just the aggregate. Run A looked fine at
  +5.3% overall while doubling the error on the calmest decile.
- `dist` is only a true reconstruction error in direct mode (t''=0). Otherwise it averages
  over random t'' and is not comparable to the `z_hat` baseline.

## Known rough edge

`eval_full.py`'s `seen?` column reports membership of the top 20%, which was meaningful
only when training was filtered. With the current recipe every decile is trained on, so
that column reads 0%/100% misleadingly and should be removed or fixed.
