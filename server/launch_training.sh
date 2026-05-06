#!/usr/bin/env bash
# =============================================================================
# server/launch_training.sh
#
# Convenience script to launch training on the Linux server.
# Always run this from the project root directory.
#
# Usage:
#   bash server/launch_training.sh --stage 1 --stored_data data/stage12_clean.hdf5
#   bash server/launch_training.sh --stage 2 --stored_data data/stage12_clean.hdf5
#
# Stage 1: single-process (python), VLA loaded with device_map="auto" across
#          all 4 GPUs (~3.5 GB each).  No torchrun needed.
# Stage 2: multi-process (torchrun), 4 DDP workers, VLA on rank 0 only.
#
# Optional overrides (passed through to main.py):
#   --epoch N           Number of training epochs
#   --batch_size N      Per-GPU batch size
#   --learning_rate F   Adam learning rate
#   --snr_db F          Channel SNR in dB
#
# Requirements:
#   torch >= 2.0
#   4 × T4 GPUs visible to CUDA
#
# =============================================================================

set -euo pipefail

# ── Activate virtual environment ───────────────────────────────────── #
VENV_DIR="$(dirname "$(realpath "$0")")/../venv"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

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

# Avoid HDF5 multiprocessing issues
export HDF5_USE_FILE_LOCKING=FALSE

# ── Detect stage from args ─────────────────────────────────────────── #
STAGE=1
PREV=""
for arg in "$@"; do
    if [ "$PREV" = "--stage" ]; then
        STAGE="$arg"
    fi
    PREV="$arg"
done

# ── Launch ────────────────────────────────────────────────────────── #
echo "[launch] Args: $*"
echo ""

if [ "$STAGE" = "1" ]; then
    echo "[launch] Stage 1 — single-process (device_map=auto across ${NUM_GPUS} GPUs)"
    python3 main.py --train "$@"
else
    # ── NCCL tuning for T4 GPUs (no NVLink) ───────────────────────── #
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=lo,eth,en
    export NCCL_DEBUG=WARN

    echo "[launch] Stage ${STAGE} — torchrun with ${NUM_GPUS} processes"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=29500 \
        main.py --train "$@"
fi