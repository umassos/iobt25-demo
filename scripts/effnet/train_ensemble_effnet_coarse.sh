#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|a100,  # Nvidia L40s
#SBATCH -t 15:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=cifar-effnet

source ~/venvs/ensemble/bin/activate
CUT_POINT=3
LR=0.005
python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse.py \
    --model efficientnet_b0 \
    --dataset torch/cifar100 \
    --opt adam \
    --resume ./output/train/ensemble-effnet-c$CUT_POINT-coarse-lr-0.005-cifar/last.pth.tar \
    --num-classes 100 \
    --num-coarse-classes 20 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --epochs 100 \
    --experiment ensemble-effnet-c$CUT_POINT-coarse-lr-$LR-cifar \
    --run-name ensemble-effnet-c$CUT_POINT-coarse-lr-$LR-cifar \
    -b 64 \
    --lr $LR\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \

LR=0.0005
python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse_finetuning.py \
    --model efficientnet_b0 \
    --dataset torch/cifar100 \
    --opt adam \
    --initial-checkpoint ./output/train/ensemble-effnet-c$CUT_POINT-coarse-lr-0.005-cifar/model_best.pth.tar \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --num-classes 100 \
    --num-coarse-classes 20 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --epochs 100 \
    --experiment fine-tune-ensemble-effnet-c$CUT_POINT-coarse-lr-$LR-cifar \
    --run-name fine-tune-ensemble-effnet-c$CUT_POINT-coarse-lr-$LR-cifar \
    -b 64 \
    --lr $LR\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \