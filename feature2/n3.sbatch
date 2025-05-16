#!/bin/bash
#SBATCH --job-name=project2
#SBATCH --account=a-large-sc
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=4
#SBATCH --time=0-00:15:00
#SBATCH --output=a
#SBATCH --error=b
#SBATCH --environment=/iopsstor/scratch/cscs/mgwozdz/ngc_pt_jan.toml

set -euo pipefail

srun python3 /iopsstor/scratch/cscs/mgwozdz/feat2/pretokenize.py \
    --input /capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet \
    --output /iopsstor/scratch/cscs/$USER/datasets/train_data_tok.parquet \
    --tokenizer unsloth/Mistral-Nemo-Base-2407-bnb-4bit \
    --sequence-length 200 \
    --batch-size 4096

exit 0
