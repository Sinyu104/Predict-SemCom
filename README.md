# Predictive Semantic Communication for Robot Control

A research implementation of a **Predictive Semantic Communication System** for bandwidth-efficient robot control over Rayleigh fading wireless channels.

---

## Core Idea

Instead of transmitting the full robot observation at every timestep, we transmit only the **innovation** — the part the receiver could not predict — using a shared world model (Predictor) running on the edge server as Wyner-Ziv side information.

The system is built on a frozen, task-fine-tuned **OpenVLA-7B** backbone. The visual encoder (ViT) and projector are reused directly; a compact latent space is learned on top.

```
Device (Robot)                              Edge Server
──────────────────────────────────          ──────────────────────────────────
x_t ──[ViT Encoder (frozen)]──► tokens
     ──[Token Encoder]──────────► z_t
     ──[JSCC Encoder]──────────► s_t
                                  │
                    ~~Rayleigh Channel~~──► ŝ_t
                                                │
                                                │          (z_{t-1}, a_{t-1}) ──[Predictor]──► ẑ_t^{pred}
                                                │                   │
                                      (ŝ_t, ẑ_t^{pred})─────────────      
                                      ──[JSCC Decoder]──► ẑ_t       
                                      ──[Token Decoder]──► tokens   
                                      ──[OpenVLA Projector]──►      
                                      ──[LLM head (frozen)]──► â_t 
```

The **Predictor** (action-conditioned V-JEPA 2 style) runs on the **receiver side only**, providing Wyner-Ziv side information. The JSCC Encoder has no access to `ẑ_t^{pred}` — rate is reduced because the receiver can already predict much of `z_t` from context.

**Rate:** `KL( q(ŝ_t|z_t) || p(ŝ_t|ẑ_t^{pred}) )` — automatically near zero when the scene is predictable, high when the scene changes unexpectedly.

---

## Hardware


| Machine         | Specs                            | Role                            |
| --------------- | -------------------------------- | ------------------------------- |
| Windows Desktop | RTX 3070 8 GB                    | Isaac Sim + data collection     |
| Linux Server    | 4× NVIDIA T4 16 GB (64 GB total) | OpenVLA-7B inference + training |


---

## System Overview


| Stage       | Name                     | Goal                                                       |
| ----------- | ------------------------ | ---------------------------------------------------------- |
| **Stage 0** | OpenVLA Fine-tuning      | Fine-tune OpenVLA-7B on Isaac Sim task using LoRA          |
| **Stage 1** | End-to-End Task Training | Train Token Encoder + Token Decoder with frozen OpenVLA    |
| **Stage 2** | Predictor Training       | Train action-conditioned world model (V-JEPA 2 style)      |
| **Stage 3** | JSCC Training            | Train JSCC Encoder, Decoder, Side Info Encoder (Wyner-Ziv) |


---

## Transport Layer

The Windows desktop and Linux server are on different subnets. Communication uses **ROS2** with a **Fast DDS Discovery Server** for cross-subnet node discovery.


| Topic           | Direction        | Content                         |
| --------------- | ---------------- | ------------------------------- |
| `/vla/request`  | Desktop → Server | JSON: base64 JPEG + instruction |
| `/vla/response` | Server → Desktop | JSON: 7-DoF action              |


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
├── models.py                  ViT Encoder, Token Encoder/Decoder, JSCC Encoder/Decoder,
│                              Side Info Encoder, Predictor, Rayleigh Channel
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

### 1b. Collect data trajectory
See ```/isaac_sim/README.md``` for collecting data on Isaac Sim.

### 2. Test the ROS2 Connection

```bash
# Linux server
export ROS_DOMAIN_ID=66
export ROS_DISCOVERY_SERVER=10.32.33.49:11811
source /opt/ros/humble/setup.bash
python server/test_ros2_server.py
```

```cmd
:: Windows desktop
set ROS_DOMAIN_ID=66
set ROS_DISCOVERY_SERVER=10.32.33.49:11811
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
<isaac_sim_root>\python.bat isaac_sim\isaac_collector.py 
    --output data\stage12_clean.hdf5 
    --num_episodes 300 --episode_length 120
```

### 6. Train the Semantic Communication Model (Linux server)

```bash
# Stage 1 — Token Encoder + Token Decoder (end-to-end task training)
bash server/launch_training.sh --stage 1 \
    --stored_data data/stage12_clean.hdf5 --epoch 50

# Stage 2 — Predictor (action-conditioned V-JEPA 2)
bash server/launch_training.sh --stage 2 \
    --stored_data data/stage12_clean.hdf5 --epoch 30

# Stage 3 — JSCC Encoder / Decoder / Side Info Encoder (Wyner-Ziv)
bash server/launch_training.sh --stage 3 \
    --stored_data data/stage12_clean.hdf5 --epoch 30
```

### 7. Run Inference / Ablation

```bash
# Sweep SNR for rate-distortion curve
for SNR in 0 5 10 15 20; do
    python main.py --inference --snr_db $SNR \
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

## References

> Bardes et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," Meta AI 2025.

> Wyner & Ziv, "The Rate-Distortion Function for Source Coding with Side Information at the Decoder," *IEEE Trans. Inf. Theory* 1976.

