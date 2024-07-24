#!/bin/bash

# Job Name and Files
#SBATCH -J 4000_1

# Job core configuration
#SBATCH -N 1
#SBATCH --exclusive

# Wall time in format Day-hour:minutes:seconds
#SBATCH --time=2-00:00:00

## Partition:short/normal and hardware constraint
#SBATCH --partition=thin

module load 2022 
module load GCC/11.3.0
module load MPICH/4.0.2-GCC-11.3.0
module load ScaLAPACK/2.2.0-gompi-2022a-fb

source /home/yuehao/anaconda3/etc/profile.d/conda.sh
conda activate infer

python randomOptimization.py
