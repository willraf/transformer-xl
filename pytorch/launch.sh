#!/bin/bash -l
#SBATCH -p ecsstudents
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wr1g20@soton.ac.uk
#SBATCH --time=04:40:00

module load conda/py3-latest
conda activate transformer-xl

bash run_enwik8_iridis.sh train --work_dir ~/xl-exp1

