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
CUT_POINT=2
LR=0.005
DATA=cifar
python3 ./3rdparty/pytorch-image-models/train_ensemble_3.py \
    --model resnet50 \
    --dataset torch/cifar100 \
    --opt adam \
    --dataset-download true \
    --num-classes 100 \
    --output /work/pi_shenoy_umass_edu/kgudipaty/output \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --epochs 200 \
    --experiment standalone-nn12-ensemble-3-resnet-c$CUT_POINT-lr-$LR-$DATA \
    --run-name standalone-nn12-ensemble-3-resnet-c$CUT_POINT-lr-$LR-$DATA \
    -b 64 \
    --lr 0.005\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \
