"""
main.py  —  Entry point for the Predictive Semantic Communication System.

IMPORTANT: For multi-GPU training use torchrun, NOT python:

    torchrun --nproc_per_node=4 main.py --train --stage 1 \\
        --stored_data data/stage12_clean.hdf5

Or use the convenience script:
    bash server/launch_training.sh --stage 1 --stored_data data/stage12_clean.hdf5

Single-GPU / CPU testing (uses OpenVLAStub):
    python main.py --train --stage 1 --use_stub \\
        --stored_data data/test.hdf5 --epoch 2 --batch_size 2

Inference (single GPU, after training):
    python main.py --inference --tau 0.05 --snr_db 10 \\
        --stored_data data/eval_disturbed.hdf5
"""

import argparse
import os
import torch

from config  import CONFIG
from trainer import (
    Stage1Trainer, Stage2Trainer, Stage3Trainer,
    init_distributed, cleanup_distributed,
)
from inference import run_inference


# ── CLI ──────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(
        description="Predictive Semantic Communication System"
    )
    p.add_argument("--train",     action="store_true")
    p.add_argument("--inference", action="store_true")

    p.add_argument("--stage",          type=int,   default=1)
    p.add_argument("--batch_size",     type=int,   default=None,
                   help="Per-GPU batch size (effective = batch_size × num_gpus)")
    p.add_argument("--epoch",          type=int,   default=None)
    p.add_argument("--learning_rate",  type=float, default=None)

    # Agent
    p.add_argument("--use_stub",  action="store_true",
                   help="Use OpenVLAStub (no GPU, no model download). "
                        "For testing only.")
    p.add_argument("--openvla_device_map", type=str, default=None,
                   help="HuggingFace device_map for OpenVLA (default: 'auto')")

    # Sparsifier
    p.add_argument("--tau_mode", type=str,   default=None,
                   choices=["fixed", "learned"])
    p.add_argument("--tau",      type=float, default=None)
    p.add_argument("--top_k",   type=int,   default=None)

    # Channel
    p.add_argument("--snr_db",  type=float, default=None)

    # Paths
    p.add_argument("--output_data_dir", type=str, default="./outputs")
    p.add_argument("--stored_data",     type=str, default=None)

    # Architecture overrides
    p.add_argument("--latent_dim", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)

    return p.parse_args()


# ── Agent builder ─────────────────────────────────────────────────────────── #

def build_agent(args, config: dict, rank: int):
    """
    Build the AI agent.  Always built on rank 0; all ranks share the same
    agent instance (it is not replicated by DDP).

    On a multi-GPU server the agent uses device_map="auto" so HuggingFace
    automatically distributes its layers across all available GPUs.
    """
    force_stub = args.use_stub or not torch.cuda.is_available()

    if force_stub:
        from openvla_agent import OpenVLAStub
        if rank == 0:
            print("[main] Using OpenVLAStub (CPU / --use_stub mode)")
        return OpenVLAStub(
            obs_channels = config["obs_channels"],
            obs_height   = config["obs_height"],
            obs_width    = config["obs_width"],
            action_dim   = config["action_dim"],
            instruction  = config["openvla_instruction"],
        )

    from openvla_agent import OpenVLAAgent

    # Use fine-tuned weights if available, else fall back to base model
    ft_dir     = config.get("openvla_finetune_dir", "")
    model_name = ft_dir if (ft_dir and os.path.isdir(ft_dir)) \
                 else config["openvla_model_name"]

    device_map = (
        args.openvla_device_map
        or config.get("openvla_device_map", "auto")
    )

    if rank == 0:
        print(f"[main] OpenVLA model : {model_name}")
        print(f"[main] device_map    : {device_map}")

    return OpenVLAAgent(
        instruction = config["openvla_instruction"],
        unnorm_key  = config.get("openvla_unnorm_key", "bridge_orig"),
        model_name  = model_name,
        device      = device_map,   # "auto" for multi-GPU
        quantize    = config.get("openvla_quantize", False),
    )


# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()

    # Apply CLI overrides to CONFIG
    if args.tau_mode      is not None: CONFIG["tau_mode"]      = args.tau_mode
    if args.tau           is not None: CONFIG["tau"]           = args.tau
    if args.top_k         is not None: CONFIG["top_k"]         = args.top_k
    if args.snr_db        is not None: CONFIG["snr_db"]        = args.snr_db
    if args.latent_dim    is not None: CONFIG["latent_dim"]    = args.latent_dim
    if args.hidden_dim    is not None: CONFIG["hidden_dim"]    = args.hidden_dim
    if args.batch_size    is not None: CONFIG["batch_size"]    = args.batch_size
    if args.epoch         is not None: CONFIG["epochs"]        = args.epoch
    if args.learning_rate is not None: CONFIG["learning_rate"] = args.learning_rate
    CONFIG["output_dir"] = args.output_data_dir

    # ── Distributed setup ─────────────────────────────────────────────── #
    rank, world_size, is_main = init_distributed()

    # Each rank gets its own GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device     = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if is_main:
        os.makedirs(args.output_data_dir, exist_ok=True)
        print(f"[main] World size : {world_size} GPU(s)")
        print(f"[main] Device     : {device}")
        print(f"[main] Effective batch : "
              f"{CONFIG['batch_size']} × {world_size} = "
              f"{CONFIG['batch_size'] * world_size}")
        print(f"[main] tau_mode   : {CONFIG['tau_mode']}  tau={CONFIG['tau']}")
        print(f"[main] snr_db     : {CONFIG['snr_db']} dB")

    data_path = (
        args.stored_data
        or CONFIG.get("default_data_path", "data/trajectories.hdf5")
    )
    out = args.output_data_dir

    # ── TRAINING ─────────────────────────────────────────────────────── #
    if args.train:
        agent = build_agent(args, CONFIG, rank)

        if args.stage == 1:
            if is_main:
                print("\n[main] ===== STAGE 1: VIB Encoder + Reshaper =====")
            Stage1Trainer(
                CONFIG, data_path, device, agent, rank, world_size
            ).train()

        elif args.stage == 2:
            ckpt = os.path.join(out, "stage1_best.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"Stage-1 checkpoint not found at '{ckpt}'. "
                    "Run --train --stage 1 first."
                )
            if is_main:
                print("\n[main] ===== STAGE 2: Predictor =====")
            Stage2Trainer(
                CONFIG, data_path, ckpt, device, agent, rank, world_size
            ).train()

        elif args.stage == 3:
            if CONFIG.get("tau_mode", "fixed") != "learned":
                if is_main:
                    print(
                        "\n[main] Stage 3 requires --tau_mode learned.\n"
                        "       For Globecom use --tau_mode fixed and sweep --tau."
                    )
                cleanup_distributed()
                return
            ckpt = os.path.join(out, "stage2_best.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"Stage-2 checkpoint not found at '{ckpt}'."
                )
            if is_main:
                print("\n[main] ===== STAGE 3: Learned tau (JSAC) =====")
            Stage3Trainer(
                CONFIG, data_path, ckpt, device, agent, rank, world_size
            ).train()

        else:
            raise ValueError(f"--stage must be 1, 2, or 3. Got {args.stage}.")

    # ── INFERENCE  (single process, no DDP needed) ───────────────────── #
    elif args.inference:
        for candidate in ["stage3_best.pt", "stage2_best.pt", "stage1_best.pt"]:
            ckpt = os.path.join(out, candidate)
            if os.path.exists(ckpt):
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found in '{out}'. Train the model first."
            )

        agent = build_agent(args, CONFIG, rank=0)
        print(
            f"\n[main] ===== INFERENCE  "
            f"tau={CONFIG['tau']}  snr={CONFIG['snr_db']}dB  "
            f"ckpt={os.path.basename(ckpt)} ====="
        )
        run_inference(CONFIG, data_path, ckpt, device, agent)

    else:
        print("[main] Specify --train or --inference.")

    cleanup_distributed()


if __name__ == "__main__":
    main()