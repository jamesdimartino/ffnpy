#!/bin/bash

#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH --mem=100G
#SBATCH --job-name=parts

source activate ffnpy

python partition.py --input_volume $JAMES/util/3monthlab.h5:labs --output_volume /hpc/mydata/james.dimartino/ffnpy/ffn/data/labparts.h5:parts --min_size 5000

conda deactivate

