"""
main.py  —  Entry point for the VAE-latent Wyner-Ziv SemCom System.

Stage 1 — Ctrl-World (first diffusion):
  Full SVD model (recommended):
    torchrun --nproc_per_node=4 main.py --train --stage 1 \\
        --svd_path stabilityai/stable-video-diffusion-img2vid \\
        --stored_data data/train.hdf5

  Stub mode (CPU / no download, for testing only):
    python main.py --train --stage 1 --use_stub --stored_data data/test.hdf5

Stage 2 — JSCC + Refinement Diffusion (second diffusion):
  torchrun --nproc_per_node=4 main.py --train --stage 2 \\
      --svd_path stabilityai/stable-video-diffusion-img2vid \\
      --stored_data data/train.hdf5

Inference:
  python main.py --inference --snr_db 10 --stored_data data/eval.hdf5
"""

import argparse
import os
import torch

from config  import CONFIG
from trainer import Stage1Trainer, Stage2Trainer, init_distributed, cleanup_distributed


# ── VAE builder ──────────────────────────────────────────────────────────── #

def build_vae(args, config: dict):
    if args.use_stub or not torch.cuda.is_available():
        if not args.use_stub:
            print("[main] WARNING: No CUDA — falling back to stub VAE (zero latents).")
        return None

    data_path = args.stored_data or config.get("default_data_path", "data/trajectories.hdf5")
    if _hdf5_has_latents(data_path):
        print("[main] HDF5 has pre-computed latents — skipping VAE load (saves ~3 GB VRAM).")
        return None

    from vae_wrapper import VAEWrapper
    return VAEWrapper(config.get("vae_model_name", "stabilityai/sd-vae-ft-mse"))


def _hdf5_has_latents(path: str) -> bool:
    import h5py
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            if "latents" in f:
                return True
            for k in list(f.keys())[:1]:
                if "latents" in f[k]:
                    return True
    except Exception:
        pass
    return False


# ── Ctrl-World builder ───────────────────────────────────────────────────── #

def build_ctrl_world(args, config: dict):
    """
    Return a CtrlWorldWrapper (full SVD) or StubCtrlWorld (lightweight stub).

    Always returns a ctrl_world object — stages 1 and 2 both need it.
    """
    num_hist = args.num_history or config.get("num_history", 6)
    num_pred = args.num_pred    or config.get("num_pred",    1)

    if args.use_stub or not torch.cuda.is_available():
        from ctrl_world_wrapper import StubCtrlWorld
        print("[main] Ctrl-World: stub mode (no GPU / no SVD download).")
        return StubCtrlWorld(
            action_dim  = config["action_dim"],
            num_history = num_hist,
            num_pred    = num_pred,
        )

    svd_path = args.svd_path or config.get("svd_model_name")
    if not svd_path:
        from ctrl_world_wrapper import StubCtrlWorld
        print("[main] WARNING: No --svd_path given. Using StubCtrlWorld.")
        return StubCtrlWorld(
            action_dim  = config["action_dim"],
            num_history = num_hist,
            num_pred    = num_pred,
        )

    from ctrl_world_wrapper import CtrlWorldWrapper
    return CtrlWorldWrapper(
        action_dim          = config["action_dim"],
        num_history         = num_hist,
        num_pred            = num_pred,
        svd_path            = svd_path,
        ctrl_ckpt           = args.ctrl_ckpt,
        freeze_unet         = True,
        finetune_cross_attn = config.get("finetune_unet_cross_attn", False),
        dtype               = torch.float16,
    )


# ── Policy builder ───────────────────────────────────────────────────────── #

def build_policy(args, config: dict, vae):
    if args.use_stub or not torch.cuda.is_available():
        from policy_agent import Pi0FastStub
        return Pi0FastStub(
            chunk_size  = config.get("action_chunk_size", 16),
            action_dim  = config["action_dim"],
            instruction = config.get("policy_instruction", ""),
        )

    backbone = config.get("policy_backbone", "stub")
    if backbone == "openvla":
        from openvla_agent import OpenVLAAgent
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
        try:
            from vla_server import _resolve_model_dir
            ft_dir     = config.get("policy_finetune_dir", "")
            model_name = _resolve_model_dir(ft_dir) if (ft_dir and os.path.isdir(ft_dir)) \
                         else config["policy_model_name"]
        except ImportError:
            model_name = config["policy_model_name"]

        vla_agent = OpenVLAAgent(
            instruction = config.get("policy_instruction", ""),
            unnorm_key  = config.get("policy_unnorm_key", "bridge_orig"),
            model_name  = model_name,
            device      = config.get("policy_device_map", "auto"),
            quantize    = config.get("policy_quantize", False),
        )
        from policy_agent import Pi0FastPolicy
        return Pi0FastPolicy(
            chunk_size  = config.get("action_chunk_size", 16),
            action_dim  = config["action_dim"],
            d_model     = config.get("policy_d_model",   512),
            n_layers    = config.get("policy_n_layers",    4),
            n_heads     = config.get("policy_n_heads",     8),
            backbone    = "openvla",
            vla_agent   = vla_agent,
            instruction = config.get("policy_instruction", ""),
        )
    else:
        from policy_agent import Pi0FastStub
        return Pi0FastStub(
            chunk_size  = config.get("action_chunk_size", 16),
            action_dim  = config["action_dim"],
            instruction = config.get("policy_instruction", ""),
        )


# ── CLI ──────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="VAE-latent Wyner-Ziv SemCom System")
    p.add_argument("--train",     action="store_true")
    p.add_argument("--inference", action="store_true")
    p.add_argument("--val_only",  action="store_true")

    p.add_argument("--stage",         type=int,   default=1)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--epoch",         type=int,   default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--snr_db",        type=float, default=None)
    p.add_argument("--D_jscc",        type=int,   default=None)

    p.add_argument("--use_stub", action="store_true",
                   help="Use stub VAE, Ctrl-World, and policy (no GPU, no downloads). For testing only.")

    p.add_argument("--svd_path",    type=str, default=None,
                   help="Path or HF hub ID for SVD pretrained weights "
                        "(e.g. stabilityai/stable-video-diffusion-img2vid). "
                        "Falls back to StubCtrlWorld if not provided.")
    p.add_argument("--ctrl_ckpt",   type=str, default=None,
                   help="CTRL-WORLD .pt checkpoint to initialise Action_encoder2 "
                        "(usually not needed — loaded from stage1_best.pt in Stage 2).")
    p.add_argument("--num_history", type=int, default=None,
                   help="History frames for Ctrl-World (default: config.num_history = 6).")
    p.add_argument("--num_pred",    type=int, default=None,
                   help="Predicted future frames per step (default: 1).")

    p.add_argument("--output_data_dir", type=str, default="./outputs")
    p.add_argument("--stored_data",     type=str, default=None)
    p.add_argument("--resume",          type=str, default=None)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()

    # CLI overrides
    if args.snr_db        is not None: CONFIG["snr_db"]        = args.snr_db
    if args.D_jscc        is not None: CONFIG["D_jscc"]        = args.D_jscc
    if args.batch_size    is not None: CONFIG["batch_size"]    = args.batch_size
    if args.epoch         is not None: CONFIG["epochs"]        = args.epoch
    if args.learning_rate is not None: CONFIG["learning_rate"] = args.learning_rate
    CONFIG["output_dir"] = args.output_data_dir

    # Sync clip_length with num_history / num_pred
    num_hist = args.num_history or CONFIG.get("num_history", 6)
    num_pred = args.num_pred    or CONFIG.get("num_pred",    1)
    CONFIG["num_history"] = num_hist
    CONFIG["num_pred"]    = num_pred
    CONFIG["clip_length"] = num_hist + num_pred

    # Reduce CUDA allocator fragmentation (especially helpful with the large SVD UNet)
    import os as _os
    if "PYTORCH_CUDA_ALLOC_CONF" not in _os.environ:
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    rank, world_size, is_main = init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        os.makedirs(args.output_data_dir, exist_ok=True)
        print(f"[main] World size : {world_size} GPU(s)  |  Device : {device}")
        print(f"[main] snr_db={CONFIG['snr_db']}dB  D_jscc={CONFIG['D_jscc']}  "
              f"num_history={num_hist}  t''={CONFIG['noise_level_second']}")

    data_path = args.stored_data or CONFIG.get("default_data_path", "data/trajectories.hdf5")
    out       = args.output_data_dir

    if args.train:
        vae        = build_vae(args, CONFIG)
        ctrl_world = build_ctrl_world(args, CONFIG)

        if args.stage == 1:
            n = ctrl_world.num_trainable_params()
            if is_main:
                print(f"\n[main] ===== STAGE 1: Ctrl-World  trainable={n/1e6:.2f}M =====")
            Stage1Trainer(
                CONFIG, data_path, device, vae, rank, world_size,
                ctrl_world   = ctrl_world,
                resume_ckpt  = args.resume,
            ).train()

        elif args.stage == 2:
            ckpt = os.path.join(out, "stage1_best.pt")
            if not os.path.exists(ckpt) and not args.use_stub:
                raise FileNotFoundError(
                    f"Stage-1 checkpoint not found at '{ckpt}'. "
                    "Run --train --stage 1 first, or pass --use_stub for testing."
                )
            if is_main:
                print("\n[main] ===== STAGE 2: JSCC + Refinement Diffusion =====")
            Stage2Trainer(
                CONFIG, data_path, ckpt, device, vae, rank, world_size,
                ctrl_world  = ctrl_world,
                resume_ckpt = args.resume,
            ).train()

        else:
            raise ValueError(f"--stage must be 1 or 2, got {args.stage}.")

    elif args.val_only:
        vae        = build_vae(args, CONFIG)
        ctrl_world = build_ctrl_world(args, CONFIG)
        ckpt_path  = args.resume or os.path.join(out, "stage1_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'")
        if is_main:
            print(f"\n[main] ===== STAGE 1 VALIDATION  ckpt={ckpt_path} =====")
        Stage1Trainer(
            CONFIG, data_path, device, vae, rank, world_size,
            ctrl_world   = ctrl_world,
            resume_ckpt  = ckpt_path,
        ).validate()

    elif args.inference:
        for candidate in ["stage2_best.pt", "stage1_best.pt"]:
            ckpt = os.path.join(out, candidate)
            if os.path.exists(ckpt):
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in '{out}'.")

        vae    = build_vae(args, CONFIG)
        policy = build_policy(args, CONFIG, vae)
        from inference import run_inference
        print(f"\n[main] ===== INFERENCE  snr={CONFIG['snr_db']}dB  ckpt={os.path.basename(ckpt)} =====")
        run_inference(CONFIG, data_path, ckpt, device, vae, policy)

    else:
        print("[main] Specify --train, --inference, or --val_only.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
