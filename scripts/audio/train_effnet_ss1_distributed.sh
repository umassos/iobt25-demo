#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memorys
#SBATCH -p gpu  # Partition
#SBATCH --gres=gpu:2  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=speech-effnet

source ~/venvs/ensemble/bin/activate


torchrun --standalone --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_effnet_audio.py \
        --model efficientnet_b0 \
        --num-classes 35 \
        --opt adam \
        --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
        --seed 42 \
        --epochs 100 \
        --experiment ss1-effnet-c4-nn1-lr-0.001-speech \
        --run-name ss1-effnet-c4-nn1-lr-0.001-speech \
        -b 64 \
        --lr 0.001 \
        --cut-point 4\

torchrun --standalone --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_effnet_audio.py \
        --model efficientnet_b0 \
        --dataset folder \
        --num-classes 35 \
        --opt adam \
        --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
        --seed 42 \
        --epochs 100 \
        --experiment ss1-effnet-c4-nn2-lr-0.001-speech \
        --run-name ss1-effnet-c4-nn2-lr-0.001-speech \
        -b 64 \
        --lr 0.001 \
        --cut-point 4 \

torchrun --standalone --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_ensemble_effnet_audio.py \
    --model efficientnet_b0 \
    --dataset folder \
    --opt adam \
    --use-contrastive false \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --dataset-download true \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
    --seed 42 \
    --checkpoint-nn1 ./output/train/ss1-effnet-c4-nn1-lr-0.001-speech/model_best.pth.tar \
    --checkpoint-nn2 ./output/train/ss1-effnet-c4-nn2-lr-0.001-speech/model_best.pth.tar \
    --epochs 100 \
    --experiment train-head-ss1-effnet-c4-lr-0.001-speech \
    --run-name train-head-ss1-effnet-c4-lr-0.001-speech \
    -b 64 \
    --lr 0.001\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point 4 \
    --head-channels 160