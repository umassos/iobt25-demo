#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|a100,  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=cifar-vit

source ~/venvs/ensemble/bin/activate
CUT_POINT=1
LR=0.0001
# python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse.py \
#     --model vit \
#     --dataset torch/cifar100 \
#     --opt adam \
#     --num-classes 100 \
#     --num-coarse-classes 20 \
#     --output /work/pi_shenoy_umass_edu/kgudipaty/output \
#     --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#     --seed 42 \
#     --epochs 150 \
#     --experiment ensemble-vit-c$CUT_POINT-coarse-lr-$LR-cifar \
#     --run-name ensemble-vit-c$CUT_POINT-coarse-lr-$LR-cifar \
#     -b 64 \
#     --lr $LR\
#     --loss-weights 1 1 1 \
#     --contrastive-eps 0 \
#     --cut-point $CUT_POINT \

LR=0.0001
python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse.py \
    --model vit \
    --dataset torch/cifar100 \
    --opt adam \
    --output /work/pi_shenoy_umass_edu/kgudipaty/output \
    --initial-checkpoint /work/pi_shenoy_umass_edu/kgudipaty/output/ensemble-vit-c$CUT_POINT-coarse-lr-0.005-cifar/model_best.pth.tar \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --num-classes 100 \
    --num-coarse-classes 20 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --epochs 100 \
    --experiment fine-tune-ensemble-vit-c$CUT_POINT-coarse-lr-$LR-cifar \
    --run-name fine-tune-ensemble-vit-c$CUT_POINT-coarse-lr-$LR-cifar \
    -b 64 \
    --lr $LR\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \