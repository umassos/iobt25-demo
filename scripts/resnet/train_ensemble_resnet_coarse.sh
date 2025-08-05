#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|l4|rtx8000|a100,  # Nvidia L40s
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=cifar-resnet

source ~/venvs/ensemble/bin/activate
CUT_POINT=2
LR=0.0005
python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse_finetuning.py \
    --model resnet50 \
    --dataset torch/cifar100 \
    --opt adam \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --initial-checkpoint ./output/train/ensemble-resnet-c$CUT_POINT-coarse-lr-0.005-cifar/model_best.pth.tar\
    --dataset-download true \
    --num-classes 100 \
    --num-coarse-classes 20 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --epochs 100 \
    --experiment fine-tune-ensemble-resnet-c$CUT_POINT-coarse-lr-$LR-cifar \
    --run-name fine-tune-ensemble-resnet-c$CUT_POINT-coarse-lr-$LR-cifar \
    -b 64 \
    --lr $LR\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \

# python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse.py \
# --model resnet50 \
# --dataset torch/cifar100 \
# --opt adam \
# --dataset-download true \
# --num-classes 100 \
# --num-coarse-classes 20 \
# --data-dir ./3rdparty/pytorch-image-models/datasets/ \
# --seed 42 \
# --epochs 100 \
# --experiment ensemble-resnet-c$CUT_POINT-coarse-lr-0.005-cifar \
# --run-name ensemble-resnet-c$CUT_POINT-coarse-lr-0.005-cifar \
# -b 64 \
# --lr 0.005\
# --loss-weights 0.45 0.45 0.1 \
# --contrastive-eps 0 \
# --cut-point $CUT_POINT \