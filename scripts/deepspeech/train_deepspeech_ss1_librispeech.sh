#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|a100  # Nvidia L40s
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=librispeech-deepspeech


source ~/venvs/ensemble/bin/activate
CUT_POINT=2
python3 ./3rdparty/pytorch-image-models/train_deepspeech.py \
    --model deepspeech2 \
    --dataset librispeech \
    --train-split 'train-clean-360'\
    --val-split 'test-clean'\
    --opt adam \
    --eval-metric wer \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/librispeech/ \
    --seed 42 \
    --epochs 100 \
    --experiment ss1-deepspeech-c$CUT_POINT-nn1-lr-0.001-librispeech \
    --run-name ss1-deepspeech-c$CUT_POINT-nn1-lr-0.001-librispeech \
    -b 64 \
    --lr 0.001\
    --loss-weights 1 1 1 \
    --cut-point $CUT_POINT \
    --contrastive-eps 0 \

python3 ./3rdparty/pytorch-image-models/train_deepspeech.py \
    --model deepspeech2 \
    --dataset librispeech \
    --opt adam \
    --train-split 'train-clean-360'\
    --val-split 'test-clean'\
    --eval-metric wer \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/librispeech/ \
    --seed 42 \
    --epochs 100 \
    --experiment ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-librispeech \
    --run-name ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-librispeech \
    -b 64 \
    --lr 0.001\
    --loss-weights 1 1 1 \
    --cut-point $CUT_POINT \
    --contrastive-eps 0 \


python3 ./3rdparty/pytorch-image-models/train_ensemble_deepspeech.py \
    --model deepspeech2 \
    --dataset librispeech \
    --opt adam \
    --train-split 'train-clean-360'\
    --val-split 'test-clean'\
    --freeze-nn1 true \
    --freeze-nn2 true \
    --eval-metric wer_comb \
    --checkpoint-nn1 ./output/train/ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-librispeech/model_best.pth.tar \
    --checkpoint-nn2 ./output/train/ss1-deepspeech-c$CUT_POINT-nn2-lr-0.001-librispeech/model_best.pth.tar \
    --use-contrastive false \
    --num-classes 35 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/librispeech/ \
    --seed 42 \
    --epochs 100 \
    --experiment train-head-ss1-deepspeech-c$CUT_POINT-lr-0.001-librispeech \
    --run-name train-head-ss1-deepspeech-c$CUT_POINT-lr-0.001-librispeech \
    -b 64 \
    --lr 0.001\
    --loss-weights 1 1 1 \
    --contrastive-eps 0 \
    --cut-point $CUT_POINT \