#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=cifar-effnet

source ~/venvs/ensemble/bin/activate
CUT_POINT=2
LR=0.005
DATA=cifar

# python3 ./3rdparty/pytorch-image-models/train_effnet.py \
#         --model efficientnet_b0 \
#         --dataset torch/cifar100 \
#         --output /work/pi_shenoy_umass_edu/kgudipaty/output \
#         --num-classes 100 \
#         --opt adam \
#         --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#         --seed 12 \
#         --epochs 100 \
#         --experiment ss1-effnet-c$CUT_POINT-nn-3 \
#         --run-name ss1-effnet-c$CUT_POINT-nn-3 \
#         -b 64 \
#         --lr 0.005 \
#         --cut-point $CUT_POINT \


# python3 ./3rdparty/pytorch-image-models/train_effnet.py \
#         --model efficientnet_b0 \
#         --dataset torch/cifar100 \
#         --num-classes 100 \
#         --opt adam \
#         --output /work/pi_shenoy_umass_edu/kgudipaty/output \
#         --data-dir ./3rdparty/pytorch-image-models/datasets/ \
#         --seed 24 \
#         --epochs 100 \
#         --experiment ss1-effnet-c$CUT_POINT-nn-2-$DATA \
#         --run-name ss1-effnet-c$CUT_POINT-nn-2-$DATA \
#         -b 64 \
#         --lr 0.005 \
#         --cut-point $CUT_POINT \

python3 ./3rdparty/pytorch-image-models/train_ensemble_3.py \
    --model efficientnet_b0 \
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
    --checkpoint-nn1 /work/pi_shenoy_umass_edu/kgudipaty/output/ss1-effnet-c$CUT_POINT-nn-1/model_best.pth.tar \
    --checkpoint-nn2 /work/pi_shenoy_umass_edu/kgudipaty/output/ss1-effnet-c$CUT_POINT-nn-2/model_best.pth.tar \
    --checkpoint-nn3 /work/pi_shenoy_umass_edu/kgudipaty/output/ss1-effnet-c$CUT_POINT-nn-3/model_best.pth.tar \
    --epochs 100 \
    --experiment train-head-3-ss1-effnet-c$CUT_POINT-lr-0.005-cifar \
    --run-name train-head-3-ss1-effnet-c$CUT_POINT-lr-0.005-cifar \
    -b 64 \
    --lr 0.005\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \
    --head-channels 160 


    # python3 ./3rdparty/pytorch-image-models/train_effnet.py \
#         --model efficientnet_b0 \
#         --dataset folder \
#         --num-classes 608 \
#         --opt adam \
#         --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
#         --seed 42 \
#         --epochs 100 \
#         --experiment ss1-effnet-c4-nn1-lr-0.005-tin \
#         --run-name ss1-effnet-c4-nn1-lr-0.005-tin \
#         -b 128 \
#         --lr 0.005 \
#         --cut-point 4 \


# python3 ./3rdparty/pytorch-image-models/train_effnet.py \
#         --model efficientnet_b0 \
#         --dataset folder \
#         --num-classes 608 \
#         --opt adam \
#         --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
#         --seed 42 \
#         --epochs 100 \
#         --experiment ss1-effnet-c6-nn2-lr-0.005-tin \
#         --run-name ss1-effnet-c6-nn2-lr-0.005-tin \
#         -b 128 \
#         --lr 0.005 \
#         --cut-point 6 \

# python3 ./3rdparty/pytorch-image-models/train_ensemble.py \
#     --model efficientnet_b0 \
#     --dataset folder \
#     --opt adam \
#     --use-contrastive false \
#     --freeze-nn1 true \
#     --freeze-nn2 true \
#     --dataset-download true \
#     --num-classes 608 \
#     --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/tiered-imagenet-2/ \
#     --seed 42 \
#     --checkpoint-nn1 ./output/train/ss1-effnet-c6-nn1-lr-0.005-tin/model_best.pth.tar \
#     --checkpoint-nn2 ./output/train/ss1-effnet-c6-nn1-lr-0.005-tin/model_best.pth.tar \
#     --epochs 100 \
#     --experiment train-head-ss1-effnet-c6-lr-0.005-tin \
#     --run-name train-head-ss1-effnet-c6-lr-0.005-tin \
#     -b 128 \
#     --lr 0.005\
#     --loss-weights 1 1 1 \
#     --contrastive-eps 0 \
#     --cut-point 6 \
#     --head-channels 160 