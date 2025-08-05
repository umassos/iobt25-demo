#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|a100  # Nvidia L40s
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=speech-deepspeech


source ~/venvs/ensemble/bin/activate
CUT_POINT=1
python3 ./3rdparty/pytorch-image-models/train_deepspeech.py \
    --model deepspeech2 \
    --dataset speech \
    --opt adam \
    --eval-metric wer \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
    --seed 42 \
    --epochs 100 \
    --experiment ss1-deepspeech-c$CUT_POINT-nn1-lr-0.001-speech-29 \
    --run-name ss1-deepspeech-c$CUT_POINT-nn1-lr-0.001-speech-29 \
    -b 64 \
    --lr 0.001\
    --loss-weights 1 1 1 \
    --cut-point $CUT_POINT \
    --contrastive-eps 0 \

python3 ./3rdparty/pytorch-image-models/train_deepspeech.py \
    --model deepspeech2 \
    --dataset speech \
    --opt adam \
    --eval-metric wer \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
    --seed 42 \
    --epochs 100 \
    --experiment ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-speech-29 \
    --run-name ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-speech-29 \
    -b 64 \
    --lr 0.001\
    --loss-weights 1 1 1 \
    --cut-point $CUT_POINT \
    --contrastive-eps 0 \


python3 ./3rdparty/pytorch-image-models/train_ensemble_deepspeech.py \
    --model deepspeech2 \
    --dataset speech \
    --opt adam \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --eval-metric wer_comb \
    --checkpoint-nn1 ./output/train/ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-speech-29/model_best.pth.tar \
    --checkpoint-nn2 ./output/train/ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-speech-29/model_best.pth.tar \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/SpeechCommands/speech_commands_v0.02 \
    --seed 42 \
    --epochs 100 \
    --experiment train-head-ss1-deepspeech-c$CUT_POINT-lr-0.001-speech-29 \
    --run-name train-head-ss1-deepspeech-c$CUT_POINT-lr-0.001-speech-29 \
    -b 64 \
    --lr 0.001\
    --loss-weights 0 0 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \