#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch-strawman-1
# --experiment ss1-effnet-c3-nn-2 \
#         --run-name ss1-effnet-c3-nn-2 \
export MLFLOW_EXPERIMENT_NAME=cifar-effnet
DATA=cifar
LR=0.005

source ~/venvs/ensemble/bin/activate
python3 ./3rdparty/pytorch-image-models/train_effnet.py \
        --model efficientnet_b1 \
        --dataset torch/cifar100 \
        --num-classes 100 \
        --opt adam \
        --output /work/pi_shenoy_umass_edu/kgudipaty/output \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --seed 42 \
        --epochs 200 \
        --experiment effnetb1-lr-$LR-$DATA \
        --run-name effnetb1-lr-$LR-$DATA \
        -b 64 \
        --lr $LR \
        --cut-point 0 \


# python3 ./3rdparty/pytorch-image-models/train_effnet.py \
#         --model efficientnet_b0 \
#         --dataset folder \
#         --num-classes 35 \
#         --opt adam \
#         --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/speech-commands-v2/ \
#         --train-split train \
#         --val-split val \
#         --seed 42 \
#         --epochs 100 \
#         --experiment effnet-lr-0.005-speech \
#         --run-name effnet-lr-0.005-speech \
#         -b 64 \
#         --lr 0.005 \
#         --cut-point 0 \
