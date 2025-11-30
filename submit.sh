#!/bin/bash
#SBATCH --job-name=hw5_ddpm_5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_v100_32gb'
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH -o logs/%x_%A.out
#SBATCH -e logs/%x_%A.err

set -euo pipefail
mkdir -p logs

PY="$(command -v python3 || command -v python)"

echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

srun "$PY" train_ddpm.py \
    --epochs 100 \
    --batch 128 \
    --T 500

echo "DDPM training finished."