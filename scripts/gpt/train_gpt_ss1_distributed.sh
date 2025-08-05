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
NUM_PROC=4
CUT_POINT=2

# torchrun --standalone --nproc_per_node=$NUM_PROC ./3rdparty/pytorch-image-models/train_nano_gpt.py \
#     --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/bookcorpus \
#     --experiment ss1-gpt-c$CUT_POINT-nn1-lr-0.0006-book \
#     --run-name ss1-gpt-c$CUT_POINT-nn1-lr-0.0006-book \
#     --block-size 512 \
#     --n-embd 512 \
#     --n-head 8 \
#     --max-iters 200000 \
#     --eval-interval 1000\
#     --vocab-size 8000 \
#     --gradient-accumulation-steps 4 \
#     --log-interval 50 \
#     -b 8 \
#     --cut-point $CUT_POINT \
#     --head-channels 160


# torchrun --standalone --nproc_per_node=$NUM_PROC ./3rdparty/pytorch-image-models/train_nano_gpt.py \
#     --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/bookcorpus \
#     --experiment ss1-gpt-c$CUT_POINT-nn2-lr-0.0006-book \
#     --run-name ss1-gpt-c$CUT_POINT-nn2-lr-0.0006-book \
#     --block-size 512 \
#     --n-embd 512 \
#     --n-head 8 \
#     --max-iters 200000 \
#     --eval-interval 1000\
#     --vocab-size 8000 \
#     --gradient-accumulation-steps 4 \
#     --log-interval 50 \
#     -b 8 \
#     --cut-point $CUT_POINT \
#     --head-channels 160

torchrun --standalone --nproc_per_node=$NUM_PROC ./3rdparty/pytorch-image-models/train_ensemble_nano_gpt.py \
    --dataset shakespeare_char \
    --init-from ss1 \
    --data-dir /work/pi_shenoy_umass_edu/kgudipaty/datasets/bookcorpus \
    --checkpoint-nn1 ./output/train/ss1-gpt-c$CUT_POINT-nn1-lr-0.0006-book \
    --checkpoint-nn2 ./output/train/ss1-gpt-c$CUT_POINT-nn2-lr-0.0006-book \
    --freeze-nn1 true \
    --freeze-nn2 true \
    --experiment train-head-ss1-gpt-c$CUT_POINT-lr-0.0006-book \
    --run-name train-head-ss1-gpt-c$CUT_POINT-lr-0.0006-book \
    --block-size 512 \
    --n-embd 512 \
    --n-head 8 \
    --max-iters 200000 \
    --eval-interval 1000\
    --vocab-size 8000 \
    --gradient-accumulation-steps 4 \
    --log-interval 50 \
    -b 8 \
    --loss-weights 0 0 1 \
    --cut-point $CUT_POINT \
    --head-channels 160