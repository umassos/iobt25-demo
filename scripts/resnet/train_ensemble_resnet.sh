#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|l4|rtx8000|a100,  # Nvidia L40s
#SBATCH -t 15:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=cifar-resnet

source ~/venvs/ensemble/bin/activate
CUT_POINT=3
LR=0.005
DATA=cifar

# python3 ./3rdparty/pytorch-image-models/train_ensemble.py \
#     --model resnet50 \
#     --dataset torch/cifar100 \
#     --opt adam \
#     --use-contrastive false \
#     --dataset-download true \
#     --num-classes 100 \
#     --resume /work/pi_shenoy_umass_edu/kgudipaty/output/ensemble-resnet-fc-256-c3-lr-0.005-cifar/last.pth.tar \
#     --head-type fc \
#     --hidden-dim 256 \
#     --head-channels 160\
#     --output /work/pi_shenoy_umass_edu/kgudipaty/output \
#     --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#     --seed 42 \
#     --epochs 100 \
#     --experiment ensemble-resnet-fc-256-c$CUT_POINT-lr-$LR-$DATA \
#     --run-name ensemble-resnet-fc-256-c$CUT_POINT-lr-$LR-$DATA \
#     -b 64 \
#     --lr $LR\
#     --loss-weights 1 1 1 \
#     --contrastive-eps 0 \
#     --cut-point $CUT_POINT \

    # python3 ./3rdparty/pytorch-image-models/train_ensemble.py \
    # --model resnet50\
    # --dataset torch/cifar100 \
    # --opt adam \
    # --dataset-download true \
    # --num-classes 100 \
    # --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    # --seed 42 \
    # --epochs 100 \
    # --experiment rebuttal-ensemble-resnet-c$CUT_POINT-lr-$LR-cifar \
    # --run-name rebuttal-ensemble-resnet-c$CUT_POINT-lr-$LR-cifar \
    # -b 64 \
    # --lr $LR\
    # --contrastive-eps 4 \
    # --use-contrastive true\
    # --loss-weights 1 1 1 \
    # --cut-point $CUT_POINT \
    # --head-channels 160 
    
    python3 ./3rdparty/pytorch-image-models/train_ensemble.py \
    --model resnet50 \
    --dataset torch/cifar100 \
    --opt adam \
    --initial-checkpoint /home/kgudipaty_umass_edu/ensemble-inference/output/train/rebuttal-ensemble-resnet-c3-lr-0.005-cifar/model_best.pth.tar \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --use-contrastive false \
    --num-classes 100 \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --seed 42 \
    --epochs 100 \
    --experiment rebuttal-fine-tune-ensemble-resnet-effnet-c5-lr-0.005-cifar \
    --run-name rebuttal-fine-tune-ensemble-resnet-effnet-c5-lr-0.005-cifar \
    -b 64 \
    --lr 0.001\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT\
    --head-channels 160 

    # python3 ./3rdparty/pytorch-image-models/train_ensemble_3.py \
#     --model resnet50 \
#     --dataset torch/cifar100 \
#     --opt adam \
#     --resume ./output/train/ensemble-3-resnet-c3-lr-0.005-cifar/last.pth.tar \
#     --dataset-download true \
#     --num-classes 100 \
#     --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#     --seed 42 \
#     --epochs 200 \
#     --experiment ensemble-3-resnet-c$CUT_POINT-lr-$LR-$DATA \
#     --run-name ensemble-3-resnet-c$CUT_POINT-lr-$LR-$DATA \
#     -b 64 \
#     --lr 0.005\
#     --loss-weights 1 1 1 \
#     --contrastive-eps 0 \
#     --cut-point $CUT_POINT \