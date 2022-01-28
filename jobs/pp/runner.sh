#!/bin/bash
#SBATCH --job-name=ProbReg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load gcc/7.3
module load miniconda
conda activate jbrophy-20210713

fold=$1
model=$2
tree_type=$3
delta=$4
gridsearch=$5
custom_dir=$6
tree_frac=$7

if [[ ! $custom_dir ]]; then
    custom_dir=''
fi

if [[ ! $tree_frac ]]; then
    tree_frac=1.0
fi

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/prediction.py \
  --dataset $dataset \
  --fold $fold \
  --model $model \
  --tree_type $tree_type \
  --delta $delta \
  --gridsearch $gridsearch \
  --custom_dir $custom_dir \
  --tree_frac $tree_frac \
