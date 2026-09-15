#!/bin/bash
#SBATCH --job-name=pi0fast_5colors
#SBATCH --account=def-vincentw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/sinyu104/logs/%x_%j.out
#SBATCH --error=/scratch/sinyu104/logs/%x_%j.err

# Multi-task LoRA fine-tune across the 5 pick-<color>-cube-to-tray datasets
# (99-100 episodes each, ~230 frames/episode avg — confirmed via h5py metadata
# scan before submitting). LoRA, not --full_finetune: the single-task full
# fine-tune run (job 2638456) overfit hard (loss 4.40->0.15 on 20 episodes);
# with 5x the tasks that risk doesn't go away, so sticking with the proven
# LoRA setup from the completed 10-epoch single-task run.
#
# --time=23:00:00 is a rough upper bound, not a measured estimate: multi-task
# uses MultiTaskDemoDataset, which reads frames from HDF5 lazily per-sample
# (no in-RAM preload like the single-task DemoDataset), and epochs_per_task=25
# (default) x 5 tasks x ~200 usable samples/episode ~= 25k samples/epoch vs the
# single-task run's 7864 -- both mean per-epoch time is unproven. Checking the
# first epoch's actual wall-clock once it lands, before trusting this budget
# for the full 10 epochs.

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
OUTPUT=$SCRATCH/outputs/pi0fast_5colors_finetuned

set -eo pipefail
cd $PROJECT

echo "[job] Starting at $(date)"
echo "[job] Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"

MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT server/pi0fast_server.py \
    --model_dir   $MODEL  \
    --tasks       pick_red_cube_to_tray pick_blue_cube_to_tray pick_yellow_cube_to_tray \
                  pick_orange_cube_to_tray pick_purple_cube_to_tray \
    --data_dir    $DATA_DIR \
    --finetune_output $OUTPUT \
    --episodes_per_task 25  \
    --epochs      10        \
    --batch_size  4         \
    --lora_rank   16        \
    --chunk_size  30

echo "[job] Done at $(date)"
