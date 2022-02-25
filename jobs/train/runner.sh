#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load gcc/7.3
module load miniconda
conda activate jbrophy-20210713

fold=$1
model_type=$2
tree_type=$3
gridsearch=$4
scoring=$5
custom_dir=$6

if [[ ! $custom_dir ]]; then
    custom_dir='default'
fi

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

if [[ $dataset = 'msd' ]]; then
    bagging_frac=0.1
elif [[ $dataset = 'wave' ]]; then
    bagging_frac=0.1
else
    bagging_frac=1.0
fi

python3 scripts/experiments/train.py \
  --dataset=${dataset} \
  --fold=${fold} \
  --model_type=${model_type} \
  --tree_type=${tree_type} \
  --gridsearch=${gridsearch} \
  --custom_dir=${custom_dir} \
  --bagging_frac=${bagging_frac} \
  --scoring=${scoring} \
