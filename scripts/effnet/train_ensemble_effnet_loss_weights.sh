#!/usr/bin/env bash

export MLFLOW_EXPERIMENT_NAME=kp-train-ensemble-effinet-c3-loss-weights

# Define an array of loss weight sets
loss_weights_sets=(
  "0.33 0.33 0.34"
  "0.25 0.25 0.5"
  "0.2 0.2 0.6"
  "0.14 0.14 0.72"
)

for h in 160; do
  for lw in "${loss_weights_sets[@]}"; do
    echo "Training with head channels: $h and loss weights: $lw"
    python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet.py \
        --model efficientnet_b0 \
        --initial-checkpoint ./output/train/kp-ensemble-effinet-c3-h$h-lw-${lw// /-}/model_best.pth.tar \
        --dataset torch/cifar100 \
        --dataset-download true \
        --opt adam \
        --num-classes 100 \
        --data-dir ./3rdparty/pytorch-image-models/datasets/ \
        --seed 42 \
        --epochs 100 \
        --experiment ft-ensemble-effinet-c3-h$h-lw-${lw// /-}/\
        --run-name ft-ensemble-effinet-c3-h$h-lw-${lw// /-}/\
        -b 64 \
        --lr 0.005 \
        --loss-weights $lw \
        --contrastive-eps 4 \
        --cut-point 3 \
        --last-channels $h
  done
done

