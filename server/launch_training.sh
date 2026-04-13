#!/usr/bin/env bash
# =============================================================================
# server/launch_training.sh
#
# Convenience script to launch multi-GPU training on the Linux server
# using torchrun.  Always run this from the project root directory.
#
# Usage:
#   bash server/launch_training.sh --stage 1 --stored_data data/stage12_clean.hdf5
#   bash server/launch_training.sh --stage 2 --stored_data data/stage12_clean.hdf5
#   bash server/launch_training.sh --stage 3 --tau_mode learned --stored_data data/stage12_clean.hdf5
#
# Optional overrides (passed through to main.py):
#   --epoch N           Number of training epochs
#   --batch_size N      Per-GPU batch size (default 16, effective 64 with 4 GPUs)
#   --learning_rate F   Adam learning rate
#   --tau F             Fixed sparsification threshold (Stage 1 & 2)
#   --snr_db F          Channel SNR in dB
#
# Requirements:
#   torch >= 2.0 with NCCL support
#   4 × T4 GPUs visible to CUDA
#
# =============================================================================

set -euo pipefail

# ── Validate we are in the right directory ─────────────────────────── #
if [ ! -f "main.py" ]; then
    echo "ERROR: Run this script from the project root (where main.py lives)."
    exit 1
fi

# ── GPU count ─────────────────────────────────────────────────────── #
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No CUDA GPUs detected."
    exit 1
fi
echo "[launch] Detected ${NUM_GPUS} GPU(s)"

# ── NCCL tuning for T4 GPUs ───────────────────────────────────────── #
# T4 GPUs do not support NVLink; force PCIe-based communication.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1       # disable InfiniBand if not present
export NCCL_SOCKET_IFNAME=eth0  # change to your network interface name
                                 # (run `ip link` to find it)
export NCCL_DEBUG=WARN          # set to INFO for verbose NCCL logging

# Avoid HDF5 multiprocessing issues
export HDF5_USE_FILE_LOCKING=FALSE

# ── Launch ────────────────────────────────────────────────────────── #
echo "[launch] Starting torchrun with ${NUM_GPUS} processes …"
echo "[launch] Args: $*"
echo ""

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    main.py --train "$@"