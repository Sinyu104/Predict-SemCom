# Predictive Semantic Communication for Robot Control

A research implementation of a **Predictive Semantic Communication System** for bandwidth-efficient robot control over Rayleigh fading wireless channels.

Targeting **IEEE Globecom 2026** and **IEEE JSAC** (Digital Twins special issue, deadline 1 May 2026).

---

## Core Idea

Instead of transmitting the full robot observation at every timestep, transmit only the **innovation** — the part the receiver could not predict — using a shared GRU-based Digital Twin (World Model) running on both the robot and the edge server.

```
Robot side                                    Edge Server (Digital Twin)
──────────────────────────────────            ──────────────────────────────────
obs_t ──[VIB Encoder]──► z_t
                           │
           ──[Predictor]──► ẑ_t ─────────────────────────────► ẑ_t
                           │                                       │
           z_t − ẑ_t ──► Δz_t                                     │
                           │                                       │
               [Sparsifier τ]──► Δz_sparse                        │
                           │                                       │
               ~~Rayleigh Channel~~──► ──[Reconstructor]──► Δẑ    │
                                                                   │
                                              ẑ_t + Δẑ ──► z_rec  │
                                                                   │
                                          [Reshaper]──► y_rec      │
                                                                   │
                                     [OpenVLA-7B (frozen)]──► a_t ◄┘
```

When the Predictor is accurate, `‖Δz‖ ≈ 0` → near-zero bits transmitted. Interference events cause innovation spikes, naturally allocating more bandwidth when semantic novelty is high.

---

## Hardware

| Machine | Specs | Role |
|---|---|---|
| Windows Desktop | RTX 3070 8 GB, IP 137.82.57.58 | Isaac Sim + data collection |
| Linux Server | 4× NVIDIA T4 16 GB (64 GB total), IP 10.32.33.49 | OpenVLA-7B inference + training |

---

## System Overview

- **Stage 0:** Fine-tune OpenVLA-7B on Franka pick-and-place demonstrations (LoRA)
- **Stage 1:** Train VIB Encoder + Reshaper with OpenVLA as the frozen task oracle
- **Stage 2:** Train GRU Predictor (the Digital Twin) on clean trajectories
- **Stage 3 (JSAC):** Learn per-dimension sparsification threshold τ

---

## Transport Layer

The Windows desktop and Linux server are on different subnets. Communication uses **ROS2** with a **Fast DDS Discovery Server** for cross-subnet node discovery.

| Topic | Direction | Content |
|---|---|---|
| `/vla/request` | Desktop → Server | JSON: base64 JPEG + instruction |
| `/vla/response` | Server → Desktop | JSON: 7-DoF action |

```bash
# Required on both machines
export ROS_DOMAIN_ID=66
export ROS_DISCOVERY_SERVER=10.32.33.41:11811
```

---

## Repository Structure

```
├── server/
│   ├── vla_server.py          ROS2 inference server + LoRA fine-tuning (Linux)
│   └── test_ros2_server.py    Standalone connection test (Linux side)
├── isaac_sim/
│   ├── isaac_collector.py     Isaac Sim data collector (Windows)
│   └── test_ros2_client.py    Standalone connection test (Windows side)
├── models.py                  VIB Encoder, Reshaper, Predictor, Sparsifier, Channel
├── trainer.py                 Stage 1/2/3 trainers with DDP
├── dataset.py                 HDF5 episodic dataset loader
├── inference.py               Evaluation metrics and figure generation
├── main.py                    CLI entry point (torchrun-compatible)
├── tests.py                   CPU unit tests (no GPU needed)
├── config.py                  Central configuration dict
└── requirements.txt
```

---

## Quick Start

### 1. Environment Setup (Linux server)

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
source /opt/ros/humble/setup.bash
```

### 2. Test the ROS2 Connection

```bash
# Linux server
export ROS_DOMAIN_ID=66
export ROS_DISCOVERY_SERVER=10.32.33.41:11811
source /opt/ros/humble/setup.bash
python server/test_ros2_server.py
```

```cmd
:: Windows desktop
set ROS_DOMAIN_ID=66
set ROS_DISCOVERY_SERVER=10.32.33.41:11811
python isaac_sim\test_ros2_client.py --num_requests 5
```

### 3. Fine-tune OpenVLA (Linux server)

```bash
python server/vla_server.py --mode finetune \
    --demo_data data/franka_demos.hdf5 \
    --model_dir openvla/openvla-7b \
    --finetune_output outputs/openvla_finetuned \
    --finetune_epochs 10 \
    --finetune_lr 2e-4
```

### 4. Start the VLA Inference Server (Linux server)

```bash
export ROS_DOMAIN_ID=66
export ROS_DISCOVERY_SERVER=10.32.33.41:11811
source /opt/ros/humble/setup.bash
source venv/bin/activate
python server/vla_server.py \
    --model_dir outputs/openvla_finetuned \
    --unnorm_key franka_isaac
```

### 5. Collect Trajectories (Windows desktop)

```cmd
set ROS_DOMAIN_ID=66
set ROS_DISCOVERY_SERVER=10.32.33.41:11811
<isaac_sim_root>\python.bat isaac_sim\isaac_collector.py ^
    --output data\stage12_clean.hdf5 ^
    --num_episodes 300 --episode_length 120
```

### 6. Train the Semantic Communication Model (Linux server)

```bash
# Stage 1 — VIB Encoder + Reshaper
bash server/launch_training.sh --stage 1 \
    --stored_data data/stage12_clean.hdf5 --epoch 50

# Stage 2 — GRU Predictor (Digital Twin)
bash server/launch_training.sh --stage 2 \
    --stored_data data/stage12_clean.hdf5 --epoch 30
```

### 7. Run Inference / Ablation

```bash
for TAU in 0.01 0.05 0.10 0.20; do
    python main.py --inference --tau $TAU --snr_db 10 \
        --stored_data data/eval_disturbed.hdf5
done
```

---

## OpenVLA Fine-tuning Notes

- **Base model:** `openvla/openvla-7b` (HuggingFace) — ~14 GB fp16, auto-sharded across 4× T4
- **Adapter:** LoRA rank 32, targeting `q_proj` and `k_proj`
- **Action tokenization:** Maps 7-DoF actions to the 256 least-used LLaMA tokens (highest vocab IDs). Do **not** use raw bin indices 0–255 — those are common tokens that fight the model's pre-training.
- **Target loss:** < 2.0 for usable robot control (random chance = ln(256) ≈ 5.55)
- **transformers version:** Must be `4.47.1` — `AutoModelForVision2Seq` was removed in 5.x

---

## Dependencies

```
torch>=2.0.0  (cu124 build recommended)
torchvision
transformers==4.47.1
peft>=0.10.0
accelerate>=0.27.0
timm>=0.9.0
tokenizers>=0.15.0
h5py>=3.8.0
numpy>=1.24.0
tensorboard>=2.13.0
Pillow>=9.5.0
tqdm
rclpy  (via ROS2 Humble: source /opt/ros/humble/setup.bash)
```

---

## Reference

> Diao et al., "Aligning Task- and Reconstruction-Oriented Communications for Edge Intelligence," *IEEE JSAC* 2025. [arXiv:2502.15472](https://arxiv.org/abs/2502.15472)
