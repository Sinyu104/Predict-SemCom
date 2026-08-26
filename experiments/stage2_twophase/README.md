# Stage 2 — two-phase training

Scripts that produced the first working Wyner-Ziv channel in this project. Kept here
because the result is not otherwise reproducible, and because the ruled-out list below
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
`z_hat` (~95% correct), so suppressing a noisy channel early is locally optimal — and
doing so zeroes the gradient that would have taught the encoder what to encode. Backprop
worked; it found a good local optimum that ignores the channel. Encoder and decoder
cannot escape it alone.

## The fix

```
Phase 1   z_hat WITHHELD from the refinement entirely (input, conditioning AND residual
          base), on the top 20% by deviation. The decoder cannot emit anything without
          decoding s~, so it must build the pathway.
Phase 2   z_hat returns, FULL distribution, 30% z_hat-dropout with NO residual base when
          dropped. Teaches WHEN to correct.
```

Phase 2 **must** use the full distribution. Running it on the top 20% only gave a working
channel (+9.7%) but the model over-corrected calm frames badly (D1 `A vs C` = -104.8%,
overall -8.3%).

Wyner-Ziv is preserved: the encoder never sees `z_hat` in either phase. `z_hat` enters
only through the SideInfoEncoder prior, i.e. the conditional rate term.

## Result (full distribution, 70,368 samples)

```
A (with s~) 0.03368    B (s~ = 0) 0.04486    C (z_hat alone) 0.04166

A vs B = +24.93%    channel contribution
A vs C = +19.16%    system vs raw prediction
99.8% of frames helped by >1%        rate ~5.8 bits/frame

decile   A vs B    A vs C
D1      +35.24%     +3.8%     (calmest)
D5      +31.25%    +19.7%
D10     +15.13%    +20.0%     (most disturbed)
```

`|dL/ds~|` held at 8e-05 -> 1.95e-04 through phase 2 and kept rising, versus 3.27e-06
when it collapses.

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

beta is not the lever: at beta=0 the encoder sends 250 bits/frame and the decoder still
ignores them. That changes the encoder's *incentive*, not the decoder's *necessity*.

## Usage

```bash
# 1. cache z_t and z_hat (sharded, one process per GPU)
for i in 0 1 2 3; do
  python precompute.py --shard $i --nshards 4 --stride 3 --out pre/shard$i.pt &
done; wait

# 2. two-phase training
python twophase.py --pre pre --out ../../outputs/stage2_twophase \
    --top 0.20 --top2 1.0 --ep1 3 --ep2 5 --drop2 0.30 --bs 16

# 3. evaluation
python eval_full.py                              # full-distribution decile table
python tiny_tdp.py                               # A/B/C by deviation group and t''
python persample.py  --cache pre/zcache.pt       # per-sample benefit distribution
python gradcheck.py  --cache pre/zcache.pt       # |dL/ds~| trained vs random init
```

Batch size 16 — the 784-token attention needs ~1.26 GB per layer at batch 64.

## Measurement traps hit during this investigation

- Group disturbed frames at the **top 5%** (`pose_prob=0.05` in the collector). Using the
  top 20% diluted the group ~4x and hid the effect.
- Training-log `rate` shows a huge spiky tail from reparameterisation noise and optimiser
  transients. It is **not** evidence of rate adaptation. Spikes landing in *adjacent*
  batches under a shuffled sampler proves they are optimiser artifacts.
- Always check the **per-sample** benefit distribution, not just group means. A mean over
  62 samples cannot show an effect concentrated in a handful of frames.
- `dist` is only a true reconstruction error in direct mode (t''=0). Otherwise it averages
  over random t'' and is not comparable to the `z_hat` baseline.

## Status

`precompute.py` and `twophase.py` are being folded into `precompute_latents.py --zhat` and
`Stage2Trainer` respectively, so the recipe runs from `main.py`. They stay here until the
integrated path is verified to reproduce the numbers above.
