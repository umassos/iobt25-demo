#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100,  # Nvidia L40s
#SBATCH -t 36:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=
export MLFLOW_EXPERIMENT_NAME=cifar-vit

source ~/venvs/ensemble/bin/activate
CUT_POINT=1
python3 ./3rdparty/pytorch-image-models/train_vit.py \
        --model vit \
        --dataset torch/cifar100 \
        --num-classes 100 \
        --opt adam \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --seed 42 \
        --epochs 150 \
        --experiment ss1-vit-c$CUT_POINT-nn-1 \
        --run-name ss1-vit-c$CUT_POINT-nn-1 \
        -b 64 \
        --lr 0.0001 \
        --cut-point $CUT_POINT \


python3 ./3rdparty/pytorch-image-models/train_vit.py \
        --model vit \
        --dataset torch/cifar100 \
        --num-classes 100  \
        --opt adam \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --seed 42 \
        --epochs 150 \
        --experiment ss1-vit-c$CUT_POINT-nn-2 \
        --run-name ss1-vit-c$CUT_POINT-nn-2 \
        -b 64 \
        --lr 0.0001 \
        --cut-point $CUT_POINT \

python3 ./3rdparty/pytorch-image-models/train_ensemble.py \
    --model vit \
    --dataset torch/cifar100 \
    --opt adam \
    --use-contrastive false \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --dataset-download true \
    --num-classes 100 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --checkpoint-nn1 ./output/train/ss1-vit-c$CUT_POINT-nn-1/model_best.pth.tar \
    --checkpoint-nn2 ./output/train/ss1-vit-c$CUT_POINT-nn-2/model_best.pth.tar \
    --epochs 150 \
    --experiment train-head-ss1-vit-c$CUT_POINT-lr-0.0001-cifar \
    --run-name train-head-ss1-vit-c$CUT_POINT-lr-0.0001-cifar \
    -b 64 \
    --lr 0.0001 \
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \
