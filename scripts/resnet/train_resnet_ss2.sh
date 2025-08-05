#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|l4|rtx8000|a100,  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=scratch-strawman-2

source ~/venvs/ensemble/bin/activate

python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_ss2.py \
    --model resnet50 \
    --dataset torch/cifar100 \
    --opt adam \
    --no-prefetcher \
    --use-contrastive false \
    --freeze-nn2 true \
    --dataset-download true \
    --num-classes 100 \
    --num-coarse-classes 20 \
    --data-dir ./3rdparty/pytorch-image-models/datasets \
    --seed 42 \
    --epochs 100 \
    --experiment ss2-resnet-c2-nn1-2 \
    --run-name ss2-resnet-c2-nn1-2 \
    -b 64 \
    --lr 0.005\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point 2 \
    --head-channels 160 

python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_ss2.py \
    --model resnet50 \
    --dataset torch/cifar100 \
    --initial-checkpoint ./output/train/ss2-resnet-c2-nn1-2/model_best.pth.tar \
    --opt adam \
    --no-prefetcher \
    --second-half true \
    --use-contrastive false \
    --freeze-nn1 true \
    --dataset-download true \
    --num-classes 100 \
    --num-coarse-classes 20 \
    --data-dir ./3rdparty/pytorch-image-models/datasets \
    --seed 42 \
    --epochs 100 \
    --experiment ss2-resnet-c2-nn2-2 \
    --run-name ss2-resnet-c2-nn2-2 \
    -b 64 \
    --lr 0.005\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point 2 \
    --head-channels 160 
