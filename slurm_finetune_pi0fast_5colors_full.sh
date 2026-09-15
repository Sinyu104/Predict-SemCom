#!/bin/bash
#SBATCH --job-name=pi0fast_5colors_full
#SBATCH --account=def-vincentw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sinyu104/logs/%x_%j.out
#SBATCH --error=/scratch/sinyu104/logs/%x_%j.err

# Multi-task FULL fine-tune (all 3.46B params, no LoRA) across the 5
# pick-<color>-cube-to-tray datasets (99-100 episodes each). Same 4xA100
# sizing as the earlier single-task full-finetune (job 2638456): fp32 AdamW
# needs params+grads+2 optimizer moments ~= 55GB total, which only fits a
# 40GB A100 when FSDP-sharded across 4 GPUs (~13.8GB/GPU), not 2
# (~27.7GB/GPU before activations -- too tight, see slurm_finetune_
# pi0fast_so101_full.sh for the full math).
#
# --time=24:00:00 is a starting guess, not a measured number: multi-task's
# MultiTaskDemoDataset reads frames from HDF5 lazily per-sample (unlike the
# single-task run's in-RAM preload) and epochs_per_task=25 x 5 tasks x ~200
# usable samples/episode ~= 25k samples/epoch, ~3.2x the single-task run's
# 7864. If epoch 1's real wall-clock says 10 epochs won't fit, resume from
# the last checkpoint_epochN with --full_finetune --resume_from (same
# pattern used for the LoRA single-task run across jobs 1802424/1900422/
# 2015121/2026885) rather than re-guessing a bigger --time upfront.

# ── Environment ────────────────────────────────────────────────────────── #
module load gcc opencv/4.12.0 arrow/25.0.0 python/3.12.4

source ~/venv/pi0fast_lerobot/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE

# ── Paths ──────────────────────────────────────────────────────────────── #
PROJECT=$HOME/projects/def-vincentw/sinyu104/Predict-SemCom
MODEL=lerobot/pi0fast-base
DATA_DIR=$SCRATCH/data
OUTPUT=$SCRATCH/outputs/pi0fast_5colors_full_finetuned

set -eo pipefail
cd $PROJECT

echo "[job] Starting at $(date)"
echo "[job] Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"

MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT server/pi0fast_server.py \
    --model_dir   $MODEL  \
    --tasks       pick_red_cube_to_tray pick_blue_cube_to_tray pick_yellow_cube_to_tray \
                  pick_orange_cube_to_tray pick_purple_cube_to_tray \
    --data_dir    $DATA_DIR \
    --finetune_output $OUTPUT \
    --full_finetune \
    --episodes_per_task 25  \
    --val_split   0.2       \
    --epochs      10        \
    --batch_size  4         \
    --chunk_size  30

echo "[job] Done at $(date)"
