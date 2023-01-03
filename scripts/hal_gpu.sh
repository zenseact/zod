#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 32
#SBATCH --time 24:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/zod/logs/%j
#SBATCH --partition zprodlow
#

singularity exec --bind /workspaces/$USER:/workspace \
  --bind /staging:/staging \
  --pwd /workspace/zod/ \
  --env PYTHONPATH=/workspace/zod/ \
  /workspaces/s0000960/zod/zod.sif \
  python3 -u $@

#
#EOF