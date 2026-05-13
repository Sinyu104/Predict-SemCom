"""
config.py  —  Central configuration for the VAE-latent Wyner-Ziv SemCom System.

Architecture overview
---------------------
  Transmitter (Robot):
    x_t  → [VAE Encoder (frozen)]  → z_t
         → [JSCC Encoder]          → s_t ~ q(s_t | z_t)
         → [Rayleigh Channel]      → s̃_t

  Receiver (Edge Server) — Stage 1  (Ctrl-World, first diffusion):
    [z̃_{t-km}, ..., z̃_{t-1}], a_t  →  SDEdit from z̃_{t-1} at t'  →  ẑ_t  (Wyner-Ziv side info)

  Receiver (Edge Server) — Stage 2  (Refinement Diffusion, second diffusion):
    ẑ_t   →  [Side Info Enc]  →  p(s_t | ẑ_t)   ← rate loss only
    (ẑ_t, s̃_t)  →  SDEdit from ẑ_t at t''  →  z̃_t  →  [VAE Decoder] → x̃_t
    x̃_t   →  [Policy]  →  â_t

Training stages
---------------
  Stage 1  :  Ctrl-World (SVD-based) — fine-tune Action_encoder2 only
               L_world = E[|| ẑ_0(x_{t'}, t', c) − z_t ||²]   (DDPM x₀-prediction)
  Stage 2  :  JSCC Encoder + Refinement Diffusion + Side Info Encoder
               L = L_distortion + β·L_rate
               L_distortion = MSE(z̃_t, z_t)   [RefinementDiffusion DDPM loss]
               L_rate       = KL( q(s_t|z_t) || p(s_t|ẑ_t) )

Multi-GPU training
------------------
  torchrun --nproc_per_node=4 main.py --train --stage 1 \\
      --svd_path stabilityai/stable-video-diffusion-img2vid --stored_data data/train.hdf5

  torchrun --nproc_per_node=4 main.py --train --stage 2 \\
      --svd_path stabilityai/stable-video-diffusion-img2vid --stored_data data/train.hdf5
"""

CONFIG = {
    # ── Observation ──────────────────────────────────────────────────── #
    "obs_channels": 3,
    "obs_height":   224,
    "obs_width":    224,

    # ── VAE (Stable Diffusion VAE — frozen) ──────────────────────────── #
    # SD-VAE downsamples 8×: 224 / 8 = 28.  Latent channels = 4.
    "C_vae":          4,
    "H_vae":         28,
    "W_vae":         28,
    "vae_model_name": "stabilityai/sd-vae-ft-mse",

    # ── Action ───────────────────────────────────────────────────────── #
    "action_dim":        7,    # Franka: [x, y, z, rx, ry, rz, gripper]
    "action_chunk_size": 16,   # π0-fast style: predict K future actions at once

    # ── JSCC / Channel ───────────────────────────────────────────────── #
    "D_jscc":      256,    # transmitted symbols per step
    "snr_db":      10.0,   # Rayleigh channel SNR in dB (sweep for ablation)

    # ── JSCC module dims ─────────────────────────────────────────────── #
    "jscc_d_hidden": 512,  # hidden dim for JsccEncoder / SideInfoEncoder MLPs

    # ── Ctrl-World (Stage 1 — first diffusion, SVD-based) ────────────── #
    "svd_model_name":          "stabilityai/stable-video-diffusion-img2vid",
    "num_history":              6,    # history frames fed to Ctrl-World
    "num_pred":                 10,   # future frames to predict
    "noise_level_first":        500,  # max DDPM timestep t' sampled during Ctrl-World training
    "ctrl_world_n_steps":       10,   # DDIM denoising steps at Ctrl-World inference (pure noise → z_hat)
    "finetune_unet_cross_attn": True, # unfreeze UNet attn2 layers to help with gripper/OOD regions

    # ── Refinement Diffusion (Stage 2 — second diffusion) ────────────── #
    "noise_level_second": 250,  # SDEdit noise level t'' < t'; controls how much to deviate from ẑ_t
    "refine_d_model":     256,  # internal dim for RefinementDiffusion denoiser transformer
    "refine_n_heads":       8,  # cross-attention heads in RefinementDiffusion
    "refine_n_layers":      4,  # number of transformer blocks in RefinementDiffusion
    "refine_n_steps":      10,  # DDIM denoising steps at inference

    # ── Clip length for Stage 1/2 training ───────────────────────────── #
    # clip_length = num_history + num_pred (auto-overridden in main.py)
    "clip_length": 16,   # 6 history + 10 pred
    "clip_stride": 4,    # sample every k-th frame; actions summed over k steps

    # ── Stage 2 loss ─────────────────────────────────────────────────── #
    "beta_rate":  0.01,   # β weighting L_rate = KL(q||p)
    "lambda_rate": 0.01,  # alias for beta_rate (backward compatibility)

    # ── Policy ───────────────────────────────────────────────────────── #
    "policy_backbone":   "stub",     # "openvla" | "stub"
    "policy_d_model":     512,
    "policy_n_layers":      4,
    "policy_n_heads":       8,
    "policy_model_name":   "openvla/openvla-7b",
    "policy_unnorm_key":   "franka_isaac",
    "policy_instruction":  "pick up the red cube and place it on the tray",
    "policy_finetune_dir": "outputs/openvla_finetuned_v3",
    "policy_device_map":   "auto",
    "policy_quantize":     False,

    # ── Optimisation ─────────────────────────────────────────────────── #
    "learning_rate": 1e-4,
    "batch_size":    2,
    "epochs":       30,

    # ── Multi-GPU (DDP) ──────────────────────────────────────────────── #
    "num_gpus":        4,
    "ddp_backend":    "nccl",
    "ddp_find_unused": True,

    # ── Misc ──────────────────────────────────────────────────────────── #
    "seed":          42,
    "num_workers":    0,      # 0 = safe with h5py; increase if SWMR enabled
    "log_interval":  10,

    # ── Paths ─────────────────────────────────────────────────────────── #
    "default_data_path": "data/trajectories.hdf5",
    "output_dir":        "./outputs",
}
