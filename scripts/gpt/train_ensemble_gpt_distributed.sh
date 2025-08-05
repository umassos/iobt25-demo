#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH --gres=gpu:2  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|a100  # Nvidia L40s
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

export MLFLOW_EXPERIMENT_NAME=book-gpt

source ~/venvs/ensemble/bin/activate
CUT_POINT=1
torchrun --standalone --nproc_per_node=2 ./3rdparty/pytorch-image-models/train_ensemble_nano_gpt.py \
    --dataset shakespeare_char \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/bookcorpus \
    --experiment standalone-nn12-ensemble-gpt-c$CUT_POINT-lr-0.0006-book \
    --run-name standalone-nn12-ensemble-gpt-c$CUT_POINT-lr-0.0006-book \
    --block-size 512 \
    --n-embd 512 \
    --n-head 8 \
    --eval-interval 1000 \
    --gradient-accumulation-steps 8 \
    --log-interval 50 \
    -b 8 \
    --vocab-size 8000 \
    --loss-weights 0 0 1 \
    --cut-point $CUT_POINT\