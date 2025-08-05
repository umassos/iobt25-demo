#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH --gres=gpu:2  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|a100  # Nvidia L40s
#SBATCH -t 15:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

export MLFLOW_EXPERIMENT_NAME=tin-effnet

source ~/venvs/ensemble/bin/activate
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

CUT_POINT=5
LR=0.005
DATA=tin
torchrun --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_ensemble.py \
    --model efficientnet_b0 \
    --dataset folder \
    --opt adam \
    --use-contrastive false\
    --num-classes 608 \
    --head-type cnn \
    --resume /work/pi_shenoy_umass_edu/kgudipaty/output/ensemble-effnet-cnn-160-c5-lr-0.005-tin/last.pth.tar \
    --output /work/pi_shenoy_umass_edu/kgudipaty/output \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
    --seed 42 \
    --epochs 100 \
    --experiment ensemble-effnet-cnn-160-c$CUT_POINT-lr-$LR-$DATA \
    --run-name ensemble-effnet-cnn-160-c$CUT_POINT-lr-$LR-$DATA \
    -b 64 \
    --lr $LR\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \
    --head-channels 160 

# source ~/venvs/ensemble/bin/activate


# torchrun --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_ensemble.py \
#     --model efficientnet_b0 \
#     --dataset folder \
#     --opt adam \
#     --use-contrastive false \
#     --num-classes 608 \
#     --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
#     --seed 42 \
#     --epochs 100 \
#     --experiment standalone-nn12-ensemble-effnet-c$CUT_POINT-lr-0.005-tin \
#     --run-name standalone-nn12-ensemble-effnet-c$CUT_POINT-lr-0.005-tin \
#     -b 128 \
#     --lr 0.005\
#     --loss-weights 0 0 1 \
#     --contrastive-eps 0 \
#     --cut-point $CUT_POINT \
#     --head-channels 160 