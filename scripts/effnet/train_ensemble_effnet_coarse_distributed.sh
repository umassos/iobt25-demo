#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH --gres=gpu:2  # Number of GPUs
#SBATCH --constraint=l40s|a100  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

export MLFLOW_EXPERIMENT_NAME=tin-effnet

source ~/venvs/ensemble/bin/activate
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
CUT_POINT=5
torchrun --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse.py \
    --model efficientnet_b1 \
    --dataset folder \
    --opt adam \
    --dataset-download true \
    --num-classes 608 \
    --num-coarse-classes 34 \
    --output /work/pi_shenoy_umass_edu/kgudipaty/output \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
    --seed 42 \
    --epochs 100 \
    --experiment ensemble-effnetb1-c$CUT_POINT-coarse-lr-0.005-tin \
    --run-name ensemble-effnetb1-c$CUT_POINT-coarse-lr-0.005-tin \
    -b 64 \
    --lr 0.005\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \


torchrun --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_ensemble_effnet_coarse.py \
    --model efficientnet_b1 \
    --dataset folder \
    --opt adam \
    --freeze-nn1 true\
    --freeze-nn2 true\
    --initial-checkpoint /work/pi_shenoy_umass_edu/kgudipaty/output/ensemble-effnetb1-c$CUT_POINT-coarse-lr-0.005-tin/model_best.pth.tar \
    --dataset-download true \
    --num-classes 608 \
    --num-coarse-classes 34 \
    --output /work/pi_shenoy_umass_edu/kgudipaty/output \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
    --seed 42 \
    --epochs 100 \
    --experiment fine-tune-ensemble-effnetb1-c$CUT_POINT-coarse-lr-0.0005-tin \
    --run-name fine-tune-ensemble-effnetb1-c$CUT_POINT-coarse-lr-0.0005-tin \
    -b 64 \
    --lr 0.0005\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \