#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=speech-deepspeech


source ~/venvs/ensemble/bin/activate
CUT_POINT=2
python3 ./3rdparty/pytorch-image-models/train_ensemble_deepspeech.py \
    --model deepspeech2 \
    --dataset speech \
    --opt adam \
    --eval-metric wer_comb \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
    --seed 42 \
    --epochs 100 \
    --experiment standalone-nn12-ensemble-deepspeech-c$CUT_POINT-lr-0.002-speech-29 \
    --run-name standalone-nn12-ensemble-deepspeech-c$CUT_POINT-lr-0.002-speech-29 \
    -b 64 \
    --lr 0.002\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \


    # --experiment ensemble-effnet-c3-lr-0.001-speech \
    # --run-name ensemble-effnet-c3-lr-0.001-speech \