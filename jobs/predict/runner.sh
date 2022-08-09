#!/bin/bash
#SBATCH --job-name=Predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load gcc/7.3
module load miniconda
conda activate jbrophy-20210713

fold=$1
model_type=$2
tree_type=$3
in_scoring=$4
out_scoring=$5
tune_delta=$6
custom_in_dir=$7
custom_out_dir=$8
cond_mean_type=$9
tree_subsample_frac=${10}
tree_subsample_order=${11}
n_jobs=${12}

if [[ ! $out_scoring ]]; then
    out_scoring=${in_scoring}
fi

if [[ ! $custom_in_dir ]]; then
    custom_in_dir='default'
fi

if [[ ! $custom_out_dir ]]; then
    custom_out_dir='default'
fi

if [[ ! $cond_mean_type ]]; then
    cond_mean_type='base'
fi

if [[ ! $tree_subsample_frac ]]; then
    tree_subsample_frac=1.0
fi

if [[ ! $tree_subsample_order ]]; then
    tree_subsample_order='random'
fi

if [[ ! $n_jobs ]]; then
    n_jobs=1
fi

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/predict.py \
  --dataset=${dataset} \
  --fold=${fold} \
  --model_type=${model_type} \
  --tree_type=${tree_type} \
  --in_scoring=${in_scoring} \
  --out_scoring=${out_scoring} \
  --tune_delta=${tune_delta} \
  --custom_in_dir=${custom_in_dir} \
  --custom_out_dir=${custom_out_dir} \
  --cond_mean_type=${cond_mean_type} \
  --tree_subsample_frac=${tree_subsample_frac} \
  --tree_subsample_order=${tree_subsample_order} \
  --n_jobs=${n_jobs} \
