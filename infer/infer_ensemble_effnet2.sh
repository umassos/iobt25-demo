#!/usr/bin/env bash

export MLFLOW_EXPERIMENT_NAME=infer
# export MLFLOW_EXPERIMENT_NAME=scratch
source ~/venvs/ensemble/bin/activate

python3 ./3rdparty/pytorch-image-models/inference_ensemble_effnet2.py \
    --model efficientnet_b0 \
    --dataset torch/cifar100 \
    --num-classes 100 \
    --checkpoint ./output/train/train-head-ss1-effnet-c5-lr-0.005-cifar/model_best.pth.tar \
    --data-dir ./3rdparty/pytorch-image-models/datasets/ \
    --run-name infer-train-head-ss1-effnet-c5-lr-0.005-cifar \
    -b 64 \
    --cut-point 5 \


