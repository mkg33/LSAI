#!/bin/bash
#SBATCH --job-name=project2
#SBATCH --account=a-large-sc
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:05:00
#SBATCH --output=c
#SBATCH --error=d
#SBATCH --environment=/iopsstor/scratch/cscs/mgwozdz/ngc_pt_jan.toml

set -euo pipefail

srun python3 /iopsstor/scratch/cscs/mgwozdz/feat2/check.py

exit 0
