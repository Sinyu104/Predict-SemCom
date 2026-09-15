#!/bin/bash
#SBATCH --job-name=pi0fast_so101
#SBATCH --account=def-vincentw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=120G
#SBATCH --time=11:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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
OUTPUT=$SCRATCH/outputs/pi0fast_so101_finetuned
RESUME=$OUTPUT/checkpoint_epoch4

set -eo pipefail
cd $PROJECT

echo "[job] Starting at $(date)"
echo "[job] Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"

# Nodes here are shared across jobs (we only request 2 of 4 GPUs), so torchrun's
# default rendezvous port (29500) can already be bound by another job's process on
# the same node -> DistNetworkError: EADDRINUSE. Derive a port from the job ID so
# concurrent jobs on a shared node don't collide.
MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT server/pi0fast_server.py \
    --model_dir   $MODEL  \
    --demo_data   $DATA   \
    --finetune_output $OUTPUT \
    --resume_from $RESUME \
    --instruction "pick up the blue cube and place it in the paper box" \
    --epochs      6        \
    --batch_size  4        \
    --lora_rank   16       \
    --chunk_size  30

echo "[job] Done at $(date)"
