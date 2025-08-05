#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100|a40,  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch-strawman-1
# --experiment ss1-effnet-c3-nn-2 \
#         --run-name ss1-effnet-c3-nn-2 \
export MLFLOW_EXPERIMENT_NAME=cifar-vit

source ~/venvs/ensemble/bin/activate
python3 ./3rdparty/pytorch-image-models/train_vit.py \
        --model vit \
        --dataset torch/cifar100 \
        --resume /home/kgudipaty_umass_edu/ensemble-inference/output/train/vit-standalone-lr-0.0001/last.pth.tar \
        --num-classes 100 \
        --workers 1 \
        --opt adam \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --cut-point 12 \
        --seed 42 \
        --epochs 300 \
        --experiment vit-standalone-lr-0.0001-resume \
        --run-name vit-standalone-lr-0.0001-resume \
        -b 128 \
        --lr 0.0001 \
