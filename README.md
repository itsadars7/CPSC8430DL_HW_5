# CPSC8430DL_HW_5
# Adarsha Neupane


## Overview
This project contains implementations of **DDPM** trained on the CIFAR-10 dataset:


The model generates 32×32 RGB images.

## File Submitted
- **`train_gans.py`** – Python implementation of DDPM model.
- **`submit.sh`** – Shell script to run the DDPM training code (train_ddpm.py). 
- **`compute_metrics.py `** – Computation of Frechet Inception Distance (FID) metrics for generated images from DDPM model against real CIFAR-10 images.

## Training
DDPM model can be trained by:
### 1. Training using Python as:
**Train DDPM**
```bash
python train_ddpm.py --epochs 100 --batch 128 --T 500
```

**Optional arguments**
```bash
--beta_start   Starting value of the linear beta schedule (default 1e-4)
--beta_end     Ending value of the linear beta schedule (default 2e-2)
--lr           Learning rate (default 2e-4)
--out          Output directory (default ./outputs)
--device       "cuda" or "cpu"
--seed         Random seed (default 42)
```
Example implementation:
```bash
python train_ddpm.py --epochs 30 --batch 256 --T 1000 --lr 0.002 --seed 52 
```

### 2. Training on HPC using submit.sh as:
**Run DDPM**
```bash
sbatch submit.sh
```


## Results

DDPM model was evaluated using 128 generated images and compared against CIFAR-10 real images using Frechet Inception Distance (FID) score.

| Model      | FID (↓)        |
|------------|----------------|
| **DDPM**  | **102.59**     |


## Running the Code

Ensure dependencies are installed:
```bash
pip install torch torchvision numpy