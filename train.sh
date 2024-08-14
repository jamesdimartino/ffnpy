#!/bin/bash

#SBATCH --job-name=FFNTraining
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --mem=100G

source activate ffnpy

python train.py --train_coords /hpc/mydata/james.dimartino/ffnpy/ffnpy/data/coord.npy --data_volumes /hpc/mydata/james.dimartino/util/scripts/3mcropped.h5:raw --label_volumes /hpc/mydata/james.dimartino/util/scripts/3mgroundtruth.h5:labels --checkpoints /hpc/mydata/james.dimartino/ffnpy/ffnpy/3monthv4/ --max_epochs 100 --max_steps 100000000 --image_mean 132 --image_stddev 71

conda deactivate
