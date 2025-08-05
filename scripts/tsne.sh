#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 1:00:00  # Job time limit
#SBATCH -o tsne-compute-%j.out  # %j = job ID

# finetuning coarse ensmeble effnet on cifar100
# export MLFLOW_EXPERIMENT_NAME=cifar-effnet
# export MLFLOW_EXPERIMENT_NAME=cifar-effnet
# export MLFLOW_EXPERIMENT_NAME=scratch

source ~/venvs/ensemble/bin/activate
CUT_POINT=5
LR=0.005
DATA=tin

# python3 ./3rdparty/pytorch-image-models/tsne.py \
#     --model efficientnet_b0_3 \
#     --dataset torch/cifar100 \
#     --opt adam \
#     --tsne-nn1 ./tsne/ensemble-3-effnet-c5-tsne-1-$DATA.npy \
#     --tsne-nn2 ./tsne/ensemble-3-effnet-c5-tsne-2-$DATA.npy \
#     --tsne-nn3 ./tsne/ensemble-3-effnet-c5-tsne-3-$DATA.npy \
#     --use-contrastive false\
#     --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#     --output ./tsne/ensemble-3-effnet-c5-tsne-$DATA.png \
#     --num-classes 100 \
#     --seed 42 \
#     -b 64 \
#     --cut-point $CUT_POINT \
#     --initial-checkpoint ./output/train/ensemble-3-effnet-c5-lr-0.005-cifar/model_best.pth.tar

# python3 ./3rdparty/pytorch-image-models/tsne.py \
#     --model efficientnet_b0 \
#     --dataset torch/cifar100 \
#     --opt adam \
#     --tsne-nn1 ./tsne/ensemble-effnet-c$CUT_POINT-tsne-1-$DATA.npy \
#     --tsne-nn2 ./tsne/ensemble-effnet-c$CUT_POINT-tsne-2-$DATA.npy \
#     --use-contrastive false\
#     --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#     --output ./tsne/ensemble-effnet-c$CUT_POINT-tsne-$DATA.png \
#     --num-classes 100 \
#     --seed 42 \
#     -b 64 \
#     --cut-point $CUT_POINT \
#     --initial-checkpoint ./output/train/ensemble-effnet-c$CUT_POINT-lr-0.005-$DATA/model_best.pth.tar

python3 ./3rdparty/pytorch-image-models/tsne.py \
    --model efficientnet_b0_ss1 \
    --dataset folder \
    --opt adam \
    --tsne-nn1 ./tsne/ss1-effnet-c$CUT_POINT-tsne-$DATA.npy \
    --tsne-nn2 ./tsne/ss1-effnet-c$CUT_POINT-tsne-$DATA.npy \
    --use-contrastive false\
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
    --output ./tsne/ss1-effnet-c$CUT_POINT-tsne-$DATA.png \
    --num-classes 608 \
    --seed 42 \
    -b 64 \
    --cut-point $CUT_POINT \
    --initial-checkpoint /work/pi_shenoy_umass_edu/kgudipaty/output/ss1-effnet-c5-nn1-lr-0.005-tin/model_best.pth.tar

#    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
# /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
# /work/pi_shenoy_umass_edu/kgudipaty/output/ss1-effnet-c5-nn-1/model_best.pth.tar