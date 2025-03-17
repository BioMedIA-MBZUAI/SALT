#!/bin/bash
#SBATCH --job-name=lora_without_decoder
#SBATCH --gres gpu:0
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12

nvidia-smi

python /home/abdelrahman.elsayed/med-cvpr/AllinonSAM/main.py
