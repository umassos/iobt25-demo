#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=v100&vram32|l40s|rtx8000|a100  # Nvidia L40s
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

# export MLFLOW_EXPERIMENT_NAME=scratch
export MLFLOW_EXPERIMENT_NAME=owt-gpt

source ~/venvs/ensemble/bin/activate
CUT_POINT=4
python3 ./3rdparty/pytorch-image-models/train_ensemble_nano_gpt.py \
    --dataset shakespeare_char \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/bookcorpus \
    --experiment ensemble-gpt-c$CUT_POINT-lr-0.0006-owt-2 \
    --run-name ensemble-gpt-c$CUT_POINT-lr-0.0006-owt-2 \
    --eval-interval 2000\
    --gradient-accumulation-steps 10 \
    --log-interval 50 \
    -b 20 \
    --loss-weights 0.33 0.33 0.34 \
    --cut-point $CUT_POINT \
    --head-channels 160


    
    # --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/openwebtext \