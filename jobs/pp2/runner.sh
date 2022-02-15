#!/bin/bash
#SBATCH --job-name=ProbReg2
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
    custom_dir='prediction'
fi

if [[ ! $tree_frac ]]; then
    tree_frac=1.0
fi

if [[ $custom_dir = 'crps' ]]; then
    scoring='crps'
else
    scoring='nll'
fi

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

if [[ $dataset = 'msd' ]]; then
    tune_frac=0.1
elif [[ $dataset = 'wave' ]]; then
    tune_frac=0.1
else
    tune_frac=1.0
fi

python3 scripts/experiments/prediction.py \
  --dataset=${dataset} \
  --fold=${fold} \
  --model=${model} \
  --tree_type=${tree_type} \
  --delta=${delta} \
  --gridsearch=${gridsearch} \
  --custom_dir=${custom_dir} \
  --tree_frac=${tree_frac} \
  --tune_frac=${tune_frac} \
  --out_dir='output2' \
  --data_dir='data_20folds' \
  --scoring=${scoring} \