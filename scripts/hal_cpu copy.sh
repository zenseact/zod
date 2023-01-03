#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task 1
#SBATCH --time 2:00:00
#SBATCH --mem-per-cpu 1G
#SBATCH --output /workspaces/%u/zod/logs/%j
#SBATCH --partition zprodcpu
#

echo ""
echo "This job was started as: $@"
echo ""

singularity exec --bind /workspaces/$USER:/workspace \
  --bind /staging:/staging \
  --pwd /workspace/zod/ \
  --env PYTHONPATH=/workspace/zod/ \
  /workspaces/s0000960/zod/zod.sif \
  $@
#
#EOF