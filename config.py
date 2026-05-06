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
    "D_vit":     2176,   # ViT patch token feature dim (DINOv2 1024 + SigLIP 1152)
    "N_patches":  256,   # Number of ViT patch tokens (16×16 for 224×224 input)
    "D_model":   4096,   # LLM hidden dimension (LLaMA-2-7B), informational only

    # ── JSCC / Predictor ─────────────────────────────────────────────── #
    "D_jscc":      512,   # JSCC channel dimension (s_t transmitted symbols)
    "action_dim":    7,   # OpenVLA: [x, y, z, rx, ry, rz, gripper]

    # ── Predictor architecture ───────────────────────────────────────── #
    # Paper spec is 1024d/24L/16H (~300M), designed for large-scale training.
    # For our single-task Franka dataset (~24K clips), a much smaller model
    # trains faster and generalises better.  Gradient only flows through the
    # predictor (frozen ViT + frozen OpenVLA), so iteration is very fast.
    #
    # Chosen: 512d / 8L / 8H  (~27M params)
    #   head_dim = 512/8 = 64; 3D-RoPE: d_dim=h_dim=w_dim=20, remainder=4.
    #
    # To scale up (e.g. on A100 with more data):
    #   "d_pred": 1024, "pred_n_layers": 24, "pred_n_heads": 16  (~307M)
    "d_pred":          512,   # predictor hidden dim
    "pred_n_layers":     8,   # transformer depth
    "pred_n_heads":      8,   # attention heads  (head_dim = 64)
    "pred_mlp_ratio":  4.0,   # GELU MLP expansion ratio
    "pred_drop_path":  0.0,   # stochastic depth; try 0.1 if overfitting
    "pose_dim":          0,   # end-effector state dim; 0 = not in dataset
    "grid_size":        16,   # sqrt(N_patches): 224/14=16 for DINOv2 patch_size=14
    "jscc_d_pred":     384,   # narrower working dim for JsccEncoder/Decoder

    # ── Clip length for Stage 1 training ─────────────────────────────── #
    # Paper: T=16 frames (4 sec at 4 fps). Memory note for 4×T4 (16 GB):
    #   T=16 → seq_len=4128, requires Flash Attention (Ampere+) or reduce T.
    #   T=4  → seq_len=1028, safe on T4 with batch_size=2.
    #   T=8  → seq_len=2056, tight but feasible with batch_size=1.
    "clip_length": 4,   # set to 16 to match paper if using A100/H100
    "clip_stride": 4,   # sample every k-th frame; actions are summed over k steps

    # ── Channel ──────────────────────────────────────────────────────── #
    "snr_db": 10.0,      # Rayleigh channel SNR in dB; sweep for ablation

    # ── Stage 3 loss ─────────────────────────────────────────────────── #
    "lambda_rate": 0.01, # λ weighting L_rate = KL(q||p) in Stage 3

    # ── Loss weights ─────────────────────────────────────────────────── #
    # Stage 1: L = L_CE(z_t,a_t) + L_CE(z_{t+1},a_{t+1}) + lambda_pred*L_pred
    "lambda_pred": 1.0,   # weight for world-model prediction loss
    "ce_freq":     10,    # compute CE loss every N batches; 1 = every batch
    "gamma_delta": 20.0,   # target scaling: model predicts γ·(z_{t+1} - z_t)
    # Legacy (kept for Stage 2 / checkpoint compat)
    "lambda_tf":   0.5,
    "lambda_roll": 0.5,

    # ── Stage 1 LoRA fine-tuning ──────────────────────────────────────── #
    "lora_r":                  32,   # LLM LoRA rank (matches original fine-tune)
    "lora_r_vit":               2,   # ViT LoRA rank (smaller = less latent shift)
    "lora_alpha":              32,
    "lora_dropout":          0.05,
    "lora_llm_target_modules": ["q_proj", "v_proj"],
    "lora_vit_target_modules": ["qkv", "proj"],   # DINOv2/SigLIP use attn.qkv / attn.proj
    # LoRA ViT gradient training requires ~3-4 GB extra activation memory per batch.
    # T4 (16 GB) has no headroom after loading the 7B model — set False.
    # Set True on A100/H100 with enough VRAM.
    "lora_train_vit": True,
    "vit_update_freq":  400,  # update ViT LoRA every N steps; predictor updates every step
    "vit_update_start": 100,  # don't update ViT LoRA until this step
    "lora_lr":               2e-5,

    # ── Phase 1 early stopping ───────────────────────────────────────── #
    "early_stop_pred":     0.98,  # stop when rolling cos > this threshold
    "early_stop_patience": 100,   # rolling window size (steps)

    # ── Optimisation ─────────────────────────────────────────────────── #
    "learning_rate": 1e-4,
    "batch_size":    4,      # per-GPU batch size (reduced for ViT LoRA grad memory)
    "phase1_epochs": 1,      # ViT LoRA + Predictor, L_pred only
    "phase2_epochs": 2,      # LLM LoRA only, L_CE only
    "epochs":       10,      # kept for compat; ignored when phase1/2_epochs are set

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
