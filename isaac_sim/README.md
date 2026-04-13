# Predictive Semantic Communication System

A PyTorch implementation of a **task-oriented JSCC system with a GRU World Model (Digital Twin)**
for bandwidth-efficient robot control over Rayleigh fading channels.

Based on the ATROC framework (Diao et al., JSAC 2025) extended with predictive latent coding
and sparse innovation transmission.

---

## Architecture

```
DEVICE (Physical Robot)                        EDGE SERVER (Digital Twin)
──────────────────────────────────────────     ────────────────────────────────────────
                                               same Predictor weights + hidden state
x_t ──[VIB Encoder]──► (μ,σ,z_t)
                              │
(z_{t-1},a_{t-1}) ──[Predictor]──► ẑ_t ─────────────────────────────────► ẑ_t
                              │                                               │
              z_t - ẑ_t ──[Subtractor]──► Δz_t                               │
                              │                                               │
                         [Sparsifier τ]──► Δz_sparse                          │
                              │                                               │
                    ~~[Rayleigh Channel]~~──► received ──[Reconstructor]──► Δẑ
                                                                              │
                                                          ẑ_t + Δẑ ──[Adder]──► z_rec
                                                                              │
                                                              [Reshaper]──► y_rec
                                                                              │
                                                         [OpenVLA (frozen)]──► a_t
```

**Key properties:**
- Only the sparse innovation `Δz_sparse` is transmitted — never the full observation
- When the Predictor is accurate, `Δz ≈ 0` → near-zero bandwidth cost
- Interference events cause `‖Δz‖` spikes → more bits transmitted automatically
- The Encoder is trained with frozen OpenVLA as the task oracle (ATROC-style VIB)

---

## File Structure

```
predictive_semcom/
├── config.py               Central hyper-parameter dictionary
├── models.py               All nn.Module classes
├── openvla_agent.py        OpenVLAAgent (real 7B) + OpenVLAStub (CPU tests)
├── trainer.py              Stage1Trainer, Stage2Trainer, Stage3Trainer
├── dataset.py              HDF5 dataset + DataLoader builder
├── inference.py            Evaluation metrics + figure generation
├── main.py                 CLI entry point
├── tests.py                Unit tests (CPU, no GPU needed)
├── requirements.txt
└── isaac_sim/
    ├── isaac_collector.py  Fine-tune OpenVLA + collect trajectories
    └── README.md
```

---

## Complete Training Workflow

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Collect human/scripted demonstrations for OpenVLA fine-tuning

Collect a small set (~50–100 episodes) of Franka pick-and-place demos
using Isaac Sim's keyboard teleoperation:

```bash
<isaac_sim_root>/python.sh -m isaacsim.examples.interactive \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device keyboard \
    --dataset_file data/franka_demos.hdf5 \
    --num_demos 50
```

### 2. Fine-tune OpenVLA on the Franka task (Stage 0)

This runs inside Isaac Sim's Python and uses LoRA for efficiency (~24 GB VRAM).

```bash
<isaac_sim_root>/python.sh isaac_sim/isaac_collector.py \
    --mode finetune \
    --demo_data data/franka_demos.hdf5 \
    --finetune_output outputs/openvla_finetuned \
    --finetune_epochs 10 \
    --headless
```

Output: `outputs/openvla_finetuned/`  (LoRA adapter weights)

### 3. Collect clean trajectories with fine-tuned OpenVLA (Stage 1 + 2 data)

```bash
<isaac_sim_root>/python.sh isaac_sim/isaac_collector.py \
    --mode collect \
    --output data/stage12_clean.hdf5 \
    --openvla_dir outputs/openvla_finetuned \
    --num_episodes 300 \
    --episode_length 120 \
    --headless
```

### 4. Collect disturbed trajectories (evaluation data)

```bash
<isaac_sim_root>/python.sh isaac_sim/isaac_collector.py \
    --mode collect \
    --output data/eval_disturbed.hdf5 \
    --openvla_dir outputs/openvla_finetuned \
    --num_episodes 100 \
    --episode_length 120 \
    --interference_action \
    --interference_pose \
    --interference_prob 0.10 \
    --headless
```

### 5. Train Stage 1 — VIB Encoder + Task-Oriented Reshaper

The Encoder is trained with **frozen OpenVLA** as the task oracle.
Loss = `L_task + β₁·KL + β₂·L_align`  (ATROC Eq. 17)

```bash
python main.py --train --stage 1 \
    --stored_data data/stage12_clean.hdf5 \
    --epoch 50 \
    --batch_size 16 \
    --output_data_dir ./outputs
```

Monitor with TensorBoard:
```bash
tensorboard --logdir outputs/tb_logs
```

Watch `Stage1/val_task` — it should decrease as the Encoder learns to
preserve only what OpenVLA needs.

### 6. Train Stage 2 — Predictor (World Model / Digital Twin)

Encoder + Reshaper are frozen. The Predictor minimises `‖Δz‖`.

```bash
python main.py --train --stage 2 \
    --stored_data data/stage12_clean.hdf5 \
    --epoch 30 \
    --output_data_dir ./outputs
```

Watch `Stage2/val_pred` — it should fall from ~0.5 toward ~0.05,
meaning the Digital Twin is accurately anticipating the next latent.

### 7. Inference — tau ablation sweep (Globecom paper figures)

```bash
for TAU in 0.01 0.05 0.10 0.20; do
    python main.py --inference \
        --tau $TAU \
        --snr_db 10 \
        --stored_data data/eval_disturbed.hdf5 \
        --output_data_dir ./outputs
done
```

SNR sweep (graceful degradation curve):

```bash
for SNR in 0 5 10 15 20; do
    python main.py --inference \
        --tau 0.05 \
        --snr_db $SNR \
        --stored_data data/eval_disturbed.hdf5 \
        --output_data_dir ./outputs
done
```

Each run saves two figures:
- `outputs/recon_tau0.05_snr10.0.png`   — original vs reconstructed grid
- `outputs/timeline_tau0.05_snr10.0.png` — ‖Δz‖, bits/step, task loss over time

### 8. (Optional) Stage 3 — Learned threshold τ (JSAC extension)

```bash
python main.py --train --stage 3 \
    --tau_mode learned \
    --stored_data data/stage12_clean.hdf5 \
    --epoch 20 \
    --output_data_dir ./outputs
```

---

## Testing Without GPU

All unit tests run on CPU using `OpenVLAStub` — no model download needed:

```bash
python tests.py
```

Expected output: 30 tests, all passing.

To test the training pipeline end-to-end on CPU with tiny data:

```bash
python main.py --train --stage 1 \
    --use_stub \
    --stored_data data/test_tiny.hdf5 \
    --epoch 2 \
    --batch_size 2 \
    --output_data_dir ./outputs_test
```

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `obs_height / obs_width` | 224 | Image size (OpenVLA expects 224×224) |
| `latent_dim` | 256 | Latent vector dimension |
| `hidden_dim` | 512 | GRU hidden size in Predictor |
| `action_dim` | 7 | OpenVLA 7-DoF action |
| `snr_db` | 10.0 | Rayleigh channel SNR in dB |
| `tau_mode` | `"fixed"` | `"fixed"` = Globecom; `"learned"` = JSAC |
| `tau` | 0.05 | Fixed sparsification threshold |
| `top_k` | `None` | Fixed budget: transmit only top-k components |
| `vib_beta1` | 1.0 | KL weight in VIB loss (Stage 1) |
| `vib_beta2` | 1.0 | Alignment weight in VIB loss (Stage 1) |
| `rollout_steps` | 5 | Multi-step rollout horizon (Stage 2) |

---

## Module Descriptions

### `JsccEncoder`  (Variational)
4× stride-2 Conv blocks → FC → `(mu, log_var, z)`.
During training: `z` sampled via reparameterisation trick so gradients
flow back from the task loss through `z` into the CNN backbone.
During inference: `z = mu` (deterministic).

### `Reshaper`  (Task-Oriented Decoder)
Mirrors the Encoder with ConvTranspose2d blocks.
Trained to produce images `y` that maximise **OpenVLA's task performance**,
not pixel-level fidelity. As a result, `y` may look different from `x`
to humans but will cause OpenVLA to predict the correct action.

### `Predictor`  (GRU World Model / Digital Twin)
Takes `(z_{t-1}, a_{t-1})` and predicts `ẑ_t`.
The same weights and synchronised hidden state run on both Device and
Edge Server, so the prediction is free — no transmission needed.

### `Sparsifier`
Suppresses components of `Δz` where `|Δz_i| ≤ τ`.
In fixed mode (Globecom): `τ` is a scalar hyperparameter swept as ablation.
In learned mode (JSAC): `τ` is a per-dimension `nn.Parameter` trained in Stage 3.

### `RayleighChannel`
`y = h·x + n, h,n ~ CN(0,1)` in real arithmetic with coherent equalization.
Zero components (suppressed by Sparsifier) remain exactly zero after the
channel, so the Adder correctly falls back to the Predictor at those dims.

### `OpenVLAAgent` / `OpenVLAStub`
`OpenVLAAgent` wraps the real 7B model with a **differentiable task-loss path**:
cross-entropy on ground-truth action tokens propagates gradients back through
`pixel_values → y_rec → Reshaper → z → Encoder`.
`OpenVLAStub` is a tiny frozen MLP with the same interface for CPU testing.

---

## Reference

> Diao et al., "Aligning Task- and Reconstruction-Oriented Communications
> for Edge Intelligence," IEEE JSAC 2025. arXiv:2502.15472

> Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model,"
> arXiv:2406.09246, 2024.
