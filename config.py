"""
config.py  —  Central configuration for the Predictive Semantic Communication System.

Deployment topology
--------------------
  Windows Desktop (RTX 3090, 8 GB)
    • Isaac Sim runs here (headless, no monitor needed in isaac_collector.py)
    • ZMQ CLIENT: sends observations, receives actions
    • Saves HDF5 trajectory files to a shared drive or SCP'd to server

  Linux Server (4 × T4 GPU)
    • vla_server.py   : ZMQ SERVER, loads OpenVLA across GPUs, serves actions
    • main.py (train) : multi-GPU training with torchrun / DDP

Network configuration
----------------------
  SERVER_HOST  : IP address of the Linux server, reachable from the desktop
  ZMQ_PORT     : port on the server that vla_server.py listens on
                 Open this port on the server firewall:
                   sudo ufw allow 5555/tcp
  ZMQ_TIMEOUT_MS : milliseconds before the desktop client raises a timeout error

Multi-GPU training
-------------------
  Launch with torchrun (not python directly):
    torchrun --nproc_per_node=4 main.py --train --stage 1 ...
  Or use server/launch_training.sh which sets all required env vars.
"""

CONFIG = {
    # ── Observation dimensions ──────────────────────────────────────── #
    "obs_channels": 3,
    "obs_height":   224,    # OpenVLA expects 224×224
    "obs_width":    224,

    # ── Latent space ────────────────────────────────────────────────── #
    "latent_dim":  256,
    "hidden_dim":  512,
    "action_dim":  7,       # OpenVLA: [x, y, z, rx, ry, rz, gripper]

    # ── VIB loss weights (Stage 1) ───────────────────────────────────── #
    "vib_beta1":  1.0,      # KL divergence weight  (rate term)
    "vib_beta2":  1.0,      # alignment task-loss weight

    # ── Stage 2 loss weights ─────────────────────────────────────────── #
    "pred_alpha": 0.5,      # end-to-end latent MSE weight
    "pred_beta":  0.1,      # task loss weight

    # ── World-model rollout ──────────────────────────────────────────── #
    "rollout_steps": 5,

    # ── Channel ─────────────────────────────────────────────────────── #
    "snr_db": 10.0,

    # ── Sparsifier ──────────────────────────────────────────────────── #
    "tau_mode": "fixed",    # "fixed" = Globecom 2026; "learned" = JSAC
    "tau":       0.05,      # sweep [0.01, 0.05, 0.10, 0.20] for ablation
    "top_k":     None,

    # Stage-3 hyperparameters (tau_mode="learned" only)
    "tau_lambda":      0.01,
    "tau_temperature": 1.0,
    "tau_anneal_rate": 0.95,
    "tau_min_temp":    0.05,

    # ── Optimisation ────────────────────────────────────────────────── #
    "learning_rate": 1e-4,
    "batch_size":    4,    # per-GPU batch size (effective = 4 × 4 = 16)
    "epochs":        50,

    # ── Multi-GPU (DDP) ─────────────────────────────────────────────── #
    # Training is launched with:
    #   torchrun --nproc_per_node=4 main.py --train --stage N ...
    # These settings are read automatically from the environment by
    # torch.distributed; you do not need to change them here.
    "num_gpus":          4,         # informational only
    "ddp_backend":       "nccl",    # NCCL for GPU-to-GPU communication
    "ddp_find_unused":   False,     # set True only if you see DDP warnings

    # ── Network (ZMQ) ───────────────────────────────────────────────── #
    # SERVER_HOST : IPv4 address of the Linux server, visible from the desktop.
    #               Example: "192.168.1.100" or a public IP if on different networks.
    # ZMQ_PORT    : TCP port that vla_server.py listens on.
    #               Must be open on the server firewall.
    # Protocol    : ZMQ REQ/REP.  Desktop sends JPEG bytes, server replies
    #               with a JSON-encoded 7-element action list.
    "server_host":       "192.168.1.100",   # ← CHANGE to your server's IP
    "zmq_port":          5555,
    "zmq_timeout_ms":    10_000,            # 10 s timeout per request
    "obs_jpeg_quality":  85,                # JPEG quality for obs compression

    # ── OpenVLA ─────────────────────────────────────────────────────── #
    # instruction and unnorm_key are now per-task, loaded from data/<task>/task.json
    "openvla_model_name":    "openvla/openvla-7b",
    "openvla_finetune_dir":  "outputs/openvla_finetuned_v3",

    # On the server, split OpenVLA across T4 GPUs using device_map="auto".
    # With 4 × 16 GB T4, the 7B model (14 GB bfloat16) fits comfortably.
    "openvla_device_map":    "auto",        # HuggingFace multi-GPU mapping
    "openvla_quantize":      False,

    # ── Fine-tuning (Stage 0) ────────────────────────────────────────── #
    "finetune_epochs":       10,
    "finetune_lr":           2e-5,
    "finetune_lora_rank":    32,
    "finetune_batch_size":   4,

    # ── Paths ────────────────────────────────────────────────────────── #
    # default_data_path is kept for backward compat with --stored_data flag.
    # For multi-task training use --tasks in main.py instead.
    "default_data_path":     "data/pick_red_cube_to_tray/demos.hdf5",
    "output_dir":            "./outputs",

    # ── Misc ─────────────────────────────────────────────────────────── #
    "seed":          42,
    "num_workers":   0,     # 0 = safe with h5py; increase if SWMR enabled
    "log_interval":  10,
}