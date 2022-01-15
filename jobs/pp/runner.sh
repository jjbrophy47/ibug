#!/bin/bash
#SBATCH --job-name=ProbReg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

fold=$1
model=$2
tree_type=$3
delta=$4

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/prediction.py \
  --dataset $dataset \
  --fold $fold \
  --model $model \
  --tree_type $tree_type \
  --delta $delta \
