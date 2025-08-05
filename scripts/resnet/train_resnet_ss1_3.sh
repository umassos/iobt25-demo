#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|l4|rtx8000|a100,  # Nvidia L40s
#SBATCH -t 20:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=cifar-resnet

source ~/venvs/ensemble/bin/activate

# python3 ./3rdparty/pytorch-image-models/train_resnet.py \
#         --model resnet_50 \
#         --dataset torch/cifar100 \
#         --num-classes 100 \
#         --opt adam \
#         --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#         --seed 42 \
#         --epochs 100 \
#         --experiment ss1-resnet-c3-nn-1 \
#         --run-name ss1-resnet-c3-nn-1 \
#         -b 64 \
#         --lr 0.005 \
#         --cut-point 3 \


# python3 ./3rdparty/pytorch-image-models/train_resnet.py \
#         --model resnet_50 \
#         --dataset torch/cifar100 \
#         --num-classes 100  \
#         --opt adam \
#         --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#         --seed 12 \
#         --epochs 100 \
#         --experiment ss1-resnet-c2-nn-3 \
#         --run-name ss1-resnet-c2-nn-3 \
#         -b 64 \
#         --lr 0.005 \
#         --cut-point 2 \

python3 ./3rdparty/pytorch-image-models/train_ensemble_3.py \
    --model resnet50 \
    --dataset torch/cifar100 \
    --opt adam \
    --use-contrastive false \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --freeze-nn3 true \
    --dataset-download true \
    --num-classes 100 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --resume ./output/train/train-head-3-ss1-resnet-c3-lr-0.005-cifar/last.pth.tar \
    --epochs 100 \
    --experiment train-head-3-ss1-resnet-c3-lr-0.005-cifar \
    --run-name train-head-3-ss1-resnet-c3-lr-0.005-cifar \
    -b 64 \
    --lr 0.005\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point 3 \


   