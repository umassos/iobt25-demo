#!/usr/bin/env bash

export MLFLOW_EXPERIMENT_NAME=ensemble-effinet-asymmetric-design


for h in 160 320 640 1280;
do
  python3 ./3rdparty/pytorch-image-models/train_ensemble_effnet_asymmetric.py \
      --model efficientnet_b0 \
      --dataset torch/cifar100 \
      --opt adam \
      --dataset-download true \
      --use-contrastive false \
      --num-classes 100 \
      --data-dir ./3rdparty/pytorch-image-models/datasets/ \
      --seed 42 \
      --epochs 100 \
      --experiment kp-ensemble-effinet-asymmetric-design-c3-adam-lr-0.005-h$h \
      --run-name kp-ensemble-effinet-asymmetric-design-c3-adam-lr-0.005-h$h\
      -b 64 \
      --lr 0.005\
      --loss-weights 1 1 1 \
      --contrastive-eps 0 \
      --cut-point-1 3 \
      --cut-point-2 4 \
      --last-channels $h
done

