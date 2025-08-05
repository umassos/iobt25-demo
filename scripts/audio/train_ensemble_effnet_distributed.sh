#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=speech-effnet

source ~/venvs/ensemble/bin/activate

torchrun --standalone --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_ensemble_effnet_audio.py \
    --model efficientnet_b0 \
    --initial-checkpoint ./output/train/ensemble-effnet-c3-lr-0.001-speech/model_best.pth.tar \
    --dataset folder \
    --opt adam \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
    --seed 42 \
    --epochs 100 \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --experiment fine-tune-ensemble-effnet-c3-lr-0.0005-speech \
    --run-name fine-tune-ensemble-effnet-c3-lr-0.0005-speech \
    -b 64 \
    --lr 0.0005\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point 3 \