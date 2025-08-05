#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|l4|rtx8000|a100,  # Nvidia L40s
#SBATCH -t 36:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=scratch-strawman-1

source ~/venvs/ensemble/bin/activate

python3 ./3rdparty/pytorch-image-models/train_resnet.py \
        --model resnet_50 \
        --dataset torch/cifar100 \
        --num-classes 100 \
        --opt adam \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --seed 42 \
        --epochs 100 \
        --experiment ss1-resnet-c3-nn-1 \
        --run-name ss1-resnet-c3-nn-1 \
        -b 64 \
        --lr 0.005 \
        --cut-point 3 \


python3 ./3rdparty/pytorch-image-models/train_resnet.py \
        --model resnet_50 \
        --dataset torch/cifar100 \
        --num-classes 100  \
        --opt adam \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --seed 42 \
        --epochs 100 \
        --experiment ss1-resnet-c3-nn-2 \
        --run-name ss1-resnet-c3-nn-2 \
        -b 64 \
        --lr 0.005 \
        --cut-point 3 \

python3 ./3rdparty/pytorch-image-models/train_ensemble.py \
    --model resnet50 \
    --dataset torch/cifar100 \
    --opt adam \
    --use-contrastive false \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --dataset-download true \
    --num-classes 100 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --checkpoint-nn1 ./output/train/ss1-resnet-c3-nn-1/model_best.pth.tar \
    --checkpoint-nn2 ./output/train/ss1-resnet-c3-nn-2/model_best.pth.tar \
    --epochs 100 \
    --experiment train-head-ss1-resnet-c3-lr-0.005-cifar \
    --run-name train-head-ss1-resnet-c3-lr-0.005-cifar \
    -b 64 \
    --lr 0.005\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point 3 \

# python3 ./3rdparty/pytorch-image-models/train_ensemble_resnet2.py \
#     --model resnet_50 \
#     --dataset torch/food101 \
#     --opt adam \
#     --dataset-download true \
#     --num-classes 100 \
#     --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#     --seed 42 \
#     --epochs 100 \
#     --experiment train_ensemble_c4_lr_0.01 \
#     --run-name train_ensemble_c4_lr_0.01 \
#     -b 64 \
#     --lr 0.005\
#     --loss-weights 1 1 1 \
#     --contrastive-eps 0 \
#     --cut-point 4 \
#     --last-channels $h 
    # --experiment train_head_ss-1_c3_lr_0.005_cifar \
    # --run-name train_head_ss-1_c3_lr_0.005_cifar \

