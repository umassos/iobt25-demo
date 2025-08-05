#!/bin/bash

#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 1:00:00  # Job time limit
#SBATCH -o train-decoders-%j.out  # %j = job ID

source ~/venvs/ensemble/bin/activate
./infer_ensemble_effnet2.sh
