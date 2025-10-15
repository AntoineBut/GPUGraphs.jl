#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=cs433
#SBATCH --output=./out.txt

module load gcc cuda/11.8.0 julia/1.10.0

julia --threads=32

