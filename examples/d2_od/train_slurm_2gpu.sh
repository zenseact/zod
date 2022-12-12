#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task 31
#SBATCH --time 120:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/agp/logs/%j.out
#SBATCH --partition ztestpreemp
#

singularity exec --nv --bind /workspaces/$USER/agp:/workspace \
  --bind /staging/dataset_donation/round_2/:/staging/dataset_donation/round_2/ \
  --pwd /workspace \
  --env PYTHONPATH=/workspace \
  /workspaces/s0000960/agp/examples/d2_od/d2.sif \
  python examples/d2_od/train.py --num-gpus 2 $@
#
#EOF