# Predictive Semantic Communication for Robot Control

A research implementation of a **Predictive Semantic Communication System** for bandwidth-efficient robot control over Rayleigh fading wireless channels.

---

## Core Idea

Instead of transmitting the full robot observation at every timestep, we transmit only the **innovation** — the part the receiver could not predict — using a shared world model (Predictor) running on the edge server as Wyner-Ziv side information.

The system uses **PI0Fast** (from LeRobot) as the VLA backbone, fine-tuned on Isaac Sim tasks via LoRA. A compact VAE latent space is learned on top for semantic compression.

```
Device (Robot)                              Edge Server
──────────────────────────────────          ──────────────────────────────────
x_t ──[VAE Encoder (frozen)]──► z_t
     ──[JSCC Encoder]──────────► s_t
                                  │
                    ~~Rayleigh Channel~~──► ŝ_t
                                                │
                                                │    (z_{t-1}, a_{t-1}) ──[Predictor]──► ẑ_t^{pred}
                                                │               │
                                      (ŝ_t, ẑ_t^{pred})────────
                                      ──[JSCC Decoder]──► ẑ_t
                                      ──[VAE Decoder]──► x̂_t
                                      ──[PI0Fast]──► â_t (chunk of 30)
```

The **Predictor** (action-conditioned world model) runs on the receiver side only, providing Wyner-Ziv side information. Rate is reduced because the receiver can already predict much of `z_t` from context.

**Rate:** `KL( q(ŝ_t|z_t) || p(ŝ_t|ẑ_t^{pred}) )` — automatically near zero when the scene is predictable, high when it changes unexpectedly.

---

## Hardware

| Machine         | Specs                            | Role                                          |
| --------------- | -------------------------------- | --------------------------------------------- |
| Windows Desktop | RTX 3070 8 GB                    | Isaac Sim + data collection + VLA client      |
| Linux Server    | 4× NVIDIA T4 16 GB (64 GB total) | PI0Fast inference + semantic comm training    |

---

## System Overview

| Stage       | Name                | Goal                                                         |
| ----------- | ------------------- | ------------------------------------------------------------ |
| **Stage 0** | PI0Fast Fine-tuning | Fine-tune PI0Fast on Isaac Sim task using LoRA (via LeRobot) |
| **Stage 1** | Predictor Training  | Train action-conditioned world model                         |
| **Stage 2** | JSCC Training       | Train JSCC Encoder, Decoder, Side Info Encoder (Wyner-Ziv)   |

---

## Transport Layer

The Windows desktop and Linux server are on different subnets. Communication uses **ROS2 Humble** with a **Fast DDS Discovery Server** for cross-subnet node discovery.

| Topic           | Direction        | Content                                          |
| --------------- | ---------------- | ------------------------------------------------ |
| `/vla/request`  | Desktop → Server | JSON: base64 JPEG + instruction + sequence ID    |
| `/vla/response` | Server → Desktop | JSON: action chunk `(30, 7)` + sequence ID       |

```bash
# Required on both machines
export ROS_DOMAIN_ID=66
export ROS_DISCOVERY_SERVER=10.32.33.49:11811
```

The Fast DDS discovery server runs on the Linux server at `10.32.33.49:11811` (UDP).

### Two-Process Server Architecture

PI0Fast requires Python 3.12 (via LeRobot), but `rclpy` only supports Python 3.10. The server therefore uses two separate processes communicating over a local TCP socket:

```
ros2_bridge.py  (Python 3.10, ROS2 sourced)
      │   /vla/request subscriber, /vla/response publisher
      │   TCP persistent connection on 127.0.0.1:5555
      ▼
vla_server.py   (Python 3.12, worldmodel conda)
      │   PI0Fast inference, action chunk generation
```

---

## Repository Structure

```
├── server/
│   ├── vla_server.py          PI0Fast TCP inference server (Python 3.12)
│   ├── ros2_bridge.py         ROS2 ↔ TCP bridge (Python 3.10)
│   ├── test_ros2_server.py    Lightweight connection test (Linux side)
│   └── launch_training.sh     Multi-GPU training launcher
├── isaac_sim/
│   ├── isaac_collector.py     Isaac Sim data collector (Windows)
│   ├── vla_runner.py          VLA evaluation runner with action chunking
│   └── test_ros2_client.py    Standalone connection test (Windows side)
├── models.py                  VAE, JSCC Encoder/Decoder, Predictor, Rayleigh Channel
├── trainer.py                 Stage 1/2/3 trainers with DDP
├── dataset.py                 HDF5 episodic dataset loader
├── inference.py               Evaluation metrics and figure generation
├── main.py                    CLI entry point (torchrun-compatible)
├── config.py                  Central configuration dict
├── vae_wrapper.py             VAE latent space utilities
└── policy_agent.py            Policy agent wrapper
```

---

## Quick Start

### 1. Collect Clean Trajectories (Windows desktop)

Use the scripted policy to collect demonstrations — no VLA needed at this stage. Activate the IsaacLab conda environment and run from the Isaac Sim root:

```cmd
conda activate env_isaaclab
cd C:\Users\sinyu104.stu\isaac-sim

python.bat Predict-SemCom\tasks\pick_orange_cube_to_tray\isaac_collector.py ^
    --scripted ^
    --num_episode 300 ^
    --episode_length 300 ^
    --camera 1 2 ^
    --headless
```

Key flags:
- `--scripted`: deterministic scripted policy (no VLA needed)
- `--camera 1 2`: enables camera sensors 1 and 2
- `--headless`: runs without the GUI renderer (faster)
- Use `--num_episode 1` for a quick sanity check

To verify the collected data, use `quick_vis.py` to preview frames from the HDF5 file:

```cmd
python.bat quick_vis.py --hdf5 data\pick_orange_cube_to_tray\demos.hdf5 --episodes 1 --camera 1 2 --frames 5
```

### 2. Fine-tune PI0Fast (Linux server)

Fine-tuning uses FSDP across 4 GPUs. LoRA weights are saved to `outputs/pi0fast_finetuned/lora_weights.pt`.

```bash
conda activate worldmodel

# Single-task fine-tune
torchrun --nproc_per_node=4 server/pi0fast_server.py \
    --model_dir lerobot/pi0fast-base \
    --demo_data data/pick_orange_cube_to_tray/demos.hdf5 \
    --finetune_output outputs/pi0fast_finetuned \
    --instruction "pick up the orange cube and place it on the tray" \
    --epochs 10 \
    --batch_size 4 \
    --lora_rank 16 \
    --chunk_size 30
```

To resume from a checkpoint:
```bash
torchrun --nproc_per_node=4 server/pi0fast_server.py \
    --resume_from outputs/pi0fast_finetuned/checkpoint_epoch5 \
    --demo_data data/pick_orange_cube_to_tray/demos.hdf5 \
    --finetune_output outputs/pi0fast_finetuned
```

### 3. Environment Setup (Linux server)

The server requires two Python environments:

**Python 3.12 — PI0Fast inference (`worldmodel` conda env):**
```bash
conda activate worldmodel
pip install lerobot
# PI0Fast base model is cached at ~/.cache/huggingface/hub/models--lerobot--pi0fast-base/
```

**Python 3.10 — ROS2 bridge (system python3):**
```bash
source ~/ros2_humble/ros2-linux/setup.bash
# rclpy is bundled with the ROS2 install
```

### 4. Test the ROS2 Connection

```bash
# Linux server — start the lightweight test server
export ROS_DOMAIN_ID=66
export ROS_DISCOVERY_SERVER=10.32.33.49:11811
source ~/ros2_humble/ros2-linux/setup.bash
python3 -u server/test_ros2_server.py
```

```cmd
:: Windows desktop — send 5 test requests
set ROS_DOMAIN_ID=66
set ROS_DISCOVERY_SERVER=10.32.33.49:11811
python isaac_sim\test_ros2_client.py --num_requests 5
```

### 5. Start the VLA Inference Server (Linux server)

The Fast DDS discovery server must be running before starting these processes.

**Terminal 1 — PI0Fast inference server (Python 3.12):**
```bash
conda activate worldmodel
python -u server/vla_server.py \
    --model_dir lerobot/pi0fast-base \
    --lora_path outputs/pi0fast_finetuned/lora_weights.pt \
    --chunk_size 30 \
    --lora_rank 16 \
    --host 127.0.0.1 \
    --port 5555
```

**Terminal 2 — ROS2 bridge (Python 3.10):**
```bash
export ROS_DOMAIN_ID=66
export ROS_DISCOVERY_SERVER=10.32.33.49:11811
source ~/ros2_humble/ros2-linux/setup.bash
python3 -u server/ros2_bridge.py --host 127.0.0.1 --port 5555
```

### 6. Run VLA Evaluation (Windows desktop)

With the inference server running on Linux (Step 5), evaluate the fine-tuned policy:

```cmd
conda activate env_isaaclab
cd C:\Users\sinyu104.stu\isaac-sim

python.bat Predict-SemCom\tasks\pick_orange_cube_to_tray\vla_runner.py ^
    --chunk_size 30 ^
    --num_episodes 50 ^
    --episode_length 300 ^
    --camera 1 2
```

### 7. Train the Semantic Communication Model (Linux server)

```bash
conda activate worldmodel
cd /path/to/Worldmodel_SC

# Stage 1 — Predictor (action-conditioned world model)
python main.py --train --stage 1 \
    --stored_data data/stage12_clean.hdf5 \
    --epoch 50

# Stage 2 — JSCC Encoder / Decoder / Side Info Encoder (Wyner-Ziv)
python main.py --train --stage 2 \
    --stored_data data/stage12_clean.hdf5 \
    --epoch 30
```

### 8. Run Inference / Ablation

```bash
for SNR in 0 5 10 15 20; do
    python main.py --inference --snr_db $SNR \
        --stored_data data/eval_disturbed.hdf5
done
```

---

## PI0Fast Notes

- **Base model:** `lerobot/pi0fast-base` (HuggingFace) — cached at `~/.cache/huggingface/hub/`
- **LoRA adapter:** rank 16, targeting `lm_head` and all Q/K/V projection layers across transformer layers
- **Action chunking:** `select_action()` called 30 times per request; full chunk `(30, 7)` returned to client
- **Python requirement:** LeRobot requires Python ≥ 3.12 — use the `worldmodel` conda env
- **ROS2 requirement:** `rclpy` C extension is Python 3.10 only — use system `python3` with ROS2 sourced

---

## Dependencies

**`worldmodel` conda env (Python 3.12):**
```
torch>=2.0.0  (CUDA build)
torchvision
lerobot        (includes PI0Fast)
h5py>=3.8.0
numpy
tensorboard
Pillow
tqdm
```

**System Python 3.10 (ROS2 bridge):**
```
rclpy          (via source ~/ros2_humble/ros2-linux/setup.bash)
```

---

## References

> Black et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control," Physical Intelligence 2024.

> Bardes et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," Meta AI 2025.

> Wyner & Ziv, "The Rate-Distortion Function for Source Coding with Side Information at the Decoder," *IEEE Trans. Inf. Theory* 1976.
