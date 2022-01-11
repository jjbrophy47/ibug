#!/bin/bash
#SBATCH --job-name=ProbReg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

model=$1
tree_type=$2
scale_bias=$3

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/prediction.py \
  --dataset $dataset \
  --model $model \
  --tree_type $tree_type \
  --scale_bias $scale_bias \
