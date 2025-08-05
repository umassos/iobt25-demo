#!/bin/bash

#SBATCH -N 1  # Number of Nodes
#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 4  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|a100  # Nvidia L40s
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch-strawman-1
# --experiment ss1-effnet-c3-nn-2 \
#         --run-name ss1-effnet-c3-nn-2 \
export MLFLOW_EXPERIMENT_NAME=book-gpt

source ~/venvs/ensemble/bin/activate
torchrun --standalone --nproc_per_node=4 ./3rdparty/pytorch-image-models/train_nano_gpt.py \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/bookcorpus \
    --experiment gpt-lr-0.0006-book \
    --run-name gpt-lr-0.0006-book \
    --vocab-size 8000 \
    --block-size 512 \
    --n-embd 512 \
    --n-head 8 \
    --eval-interval 1000 \
    --gradient-accumulation-steps 4 \
    --log-interval 50 \
    -b 8 \
    --cut-point 0 \
    --head-channels 160