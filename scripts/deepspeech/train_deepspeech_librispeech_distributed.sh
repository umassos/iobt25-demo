#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH --gres=gpu:2  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

export MLFLOW_EXPERIMENT_NAME=scratch
# export MLFLOW_EXPERIMENT_NAME=librispeech-deepspeech


source ~/venvs/ensemble/bin/activate
nvidia-smi
torchrun --standalone --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_deepspeech.py \
    --model deepspeech2 \
    --dataset librispeech \
    --opt adam \
    --train-split 'train-clean-100'\
    --val-split 'test-clean'\
    --eval-metric wer \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/librispeech/ \
    --seed 42 \
    --epochs 100 \
    --experiment deepspeech-lr-0.01-librispeech \
    --run-name deepspeech-lr-0.01-librispeech \
    -b 64 \
    --lr 0.01 \