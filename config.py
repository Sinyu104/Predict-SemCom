"""
config.py  —  Central configuration for the Predictive Semantic Communication System.

Deployment topology
--------------------
  Windows Desktop (RTX 3070)
    • Isaac Sim (headless)
    • ROS2 CLIENT: sends observations, receives actions
    • Saves HDF5 trajectory files

  Linux Server (4 × T4 GPU)
    • vla_server.py   : ROS2 SERVER, loads OpenVLA, serves actions
    • main.py (train) : multi-GPU training with torchrun / DDP

Multi-GPU training
-------------------
  torchrun --nproc_per_node=4 main.py --train --stage 1 --stored_data data/train.hdf5
"""

CONFIG = {
    # ── Observation ──────────────────────────────────────────────────── #
    "obs_channels": 3,
    "obs_height":   224,    # OpenVLA expects 224×224
    "obs_width":    224,

    # ── OpenVLA backbone dimensions ──────────────────────────────────── #
    # These must match the loaded OpenVLA model.
    # Verify by calling agent.encode_image(x) and checking output shape.
    "D_vit":     2048,   # ViT patch token feature dim (DINOv2 + SigLIP combined)
    "N_patches":  256,   # Number of ViT patch tokens (16×16 for 224×224 input)
    "D_model":   4096,   # LLM hidden dimension (LLaMA-2-7B), informational only

    # ── Latent / JSCC ────────────────────────────────────────────────── #
    "latent_dim": 512,   # TokenEncoder / TokenDecoder bottleneck
    "D_jscc":     256,   # JSCC channel dimension (s_t transmitted symbols)
    "hidden_dim": 512,   # MLP hidden size shared across TokenEncoder, Predictor, etc.
    "action_dim":   7,   # OpenVLA: [x, y, z, rx, ry, rz, gripper]

    # ── Channel ──────────────────────────────────────────────────────── #
    "snr_db": 10.0,      # Rayleigh channel SNR in dB; sweep for ablation

    # ── Stage 3 loss ─────────────────────────────────────────────────── #
    "lambda_rate": 0.01, # λ weighting L_rate = KL(q||p) in Stage 3

    # ── Optimisation ─────────────────────────────────────────────────── #
    "learning_rate": 1e-4,
    "batch_size":    2,      # per-GPU batch size
    "epochs":       10,

    # ── Multi-GPU (DDP) ──────────────────────────────────────────────── #
    "num_gpus":        4,
    "ddp_backend":    "nccl",
    "ddp_find_unused": False,

    # ── Network (ZMQ) ────────────────────────────────────────────────── #
    "server_host":      "192.168.1.100",
    "zmq_port":          5555,
    "zmq_timeout_ms":   10_000,
    "obs_jpeg_quality":    85,

    # ── OpenVLA ──────────────────────────────────────────────────────── #
    "openvla_model_name":   "openvla/openvla-7b",
    "openvla_instruction":  "pick up the red cube and place it on the tray",
    "openvla_unnorm_key":   "franka_isaac",
    "openvla_finetune_dir": "outputs/openvla_finetuned_v3",
    "openvla_device_map":   "auto",
    "openvla_quantize":     False,

    # ── Fine-tuning (Stage 0, already completed) ─────────────────────── #
    "finetune_epochs":     10,
    "finetune_lr":        2e-5,
    "finetune_lora_rank":  32,
    "finetune_batch_size":  4,

    # ── Paths ─────────────────────────────────────────────────────────── #
    "default_data_path": "data/trajectories.hdf5",
    "output_dir":        "./outputs",

    # ── Misc ──────────────────────────────────────────────────────────── #
    "seed":         42,
    "num_workers":   0,    # 0 = safe with h5py; increase if SWMR enabled
    "log_interval": 10,
}
