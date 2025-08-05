#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=librispeech-deepspeech


source ~/venvs/ensemble/bin/activate
CUT_POINT=4
python3 ./3rdparty/pytorch-image-models/train_ensemble_deepspeech.py \
    --model deepspeech2 \
    --dataset librispeech \
    --opt adam \
    --train-split 'train-clean-360'\
    --val-split 'test-clean'\
    --eval-metric wer_comb \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/librispeech/ \
    --seed 42 \
    --epochs 100 \
    --experiment ensemble-deepspeech-c$CUT_POINT-lr-0.001-librispeech \
    --run-name ensemble-deepspeech-c$CUT_POINT-lr-0.001-librispeech \
    -b 64 \
    --lr 0.001\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \

