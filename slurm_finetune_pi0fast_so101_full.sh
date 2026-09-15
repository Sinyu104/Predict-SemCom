#!/bin/bash
#SBATCH --job-name=pi0fast_so101_full
#SBATCH --account=def-vincentw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=300G
#SBATCH --time=11:00:00
#SBATCH --output=/scratch/sinyu104/logs/%x_%j.out
#SBATCH --error=/scratch/sinyu104/logs/%x_%j.err
# Logs go to $SCRATCH, not the repo dir: /project/def-vincentw is currently
# over its group quota (another user, ~1976GB of the 1000GB cap) and flagged
# by `lfs quota` as exceeding the hard limit. Writes there still succeed as
# of this writing, but keeping job logs on $SCRATCH (separate, healthy quota)
# means a future EDQUOT on /project can't cost us this job's output/error log.

# ── Why 4 GPUs, not 2 ──────────────────────────────────────────────────── #
# Narval's A100 is the 40GB SXM4 variant (confirmed: 159 nodes, 4x A100SXM4
# 40GB, 48 cores, 498GB RAM per node — Alliance node-characteristics table).
# Full fine-tune (no LoRA) needs fp32 AdamW state for all 3,457,816,304 params:
#   params 13.8GB + grads 13.8GB + 2x optimizer moments 27.6GB ≈ 55.3GB total,
#   FSDP FULL_SHARD-ed across the GPU count:
#     2 GPUs -> ~27.7GB/GPU of 40GB (69%) before activations -> too tight
#     4 GPUs -> ~13.8GB/GPU of 40GB (35%) before activations -> safe margin
# Requesting all 4 GPUs also means the node isn't shared with another job,
# so the MASTER_PORT collision this repo hit before (see slurm_finetune_
# pi0fast_so101.sh) can't happen here regardless of the derived-port fix.

# ── Environment ────────────────────────────────────────────────────────── #
# NOTE: on Narval, opencv/arrow modules must be loaded BEFORE the venv is
# activated (they inject real packages into site-packages that satisfy
# lerobot's opencv-python-headless / pyarrow requirements without a PyPI build).
module load gcc opencv/4.12.0 arrow/25.0.0 python/3.12.4

source ~/venv/pi0fast_lerobot/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE

# ── Paths ──────────────────────────────────────────────────────────────── #
PROJECT=$HOME/projects/def-vincentw/sinyu104/Predict-SemCom
MODEL=lerobot/pi0fast-base
DATA=$SCRATCH/data/pick_blue_cube_paper_box/demos.hdf5
OUTPUT=$SCRATCH/outputs/pi0fast_so101_full_finetuned
# No RESUME on the first run — full-finetune checkpoints (full_weights.pt) are
# a different format from the LoRA run's lora_weights.pt, so this starts fresh
# from $MODEL rather than continuing pi0fast_so101_finetuned/checkpoint_epoch*.

set -eo pipefail
cd $PROJECT

echo "[job] Starting at $(date)"
echo "[job] Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"

# Job-specific rendezvous port (see comment above on why this node shouldn't
# be shared anyway — kept as a harmless second layer of defense).
MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT server/pi0fast_server.py \
    --model_dir   $MODEL  \
    --demo_data   $DATA   \
    --finetune_output $OUTPUT \
    --full_finetune \
    --instruction "pick up the blue cube and place it in the paper box" \
    --epochs      10       \
    --batch_size  4        \
    --chunk_size  30

echo "[job] Done at $(date)"
