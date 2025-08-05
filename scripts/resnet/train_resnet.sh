#!/usr/bin/env bash


export MLFLOW_EXPERIMENT_NAME=cifar-resnet

source ~/venvs/ensemble/bin/activate
python3 ./3rdparty/pytorch-image-models/train_resnet.py \
        --model resnet50 \
        --dataset torch/cifar100 \
        --num-classes 100 \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --seed 42 \
        --epochs 100 \
        --experiment resnet-c2-h$h \
        --run-name resnet-c2-h$h \
        -b 64 \
        --lr 0.05 \
        --cut-point -1\
        --head-channels 160 \
