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
custom_out_dir=$7
tree_subsample_frac=$8
tree_subsample_order=$9

if [[ ! $out_scoring ]]; then
    out_scoring=${in_scoring}
fi

if [[ ! $custom_out_dir ]]; then
    custom_out_dir='default'
fi

if [[ ! $tree_subsample_frac ]]; then
    tree_subsample_frac=1.0
fi

if [[ ! $tree_subsample_order ]]; then
    tree_subsample_order='random'
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
  --custom_out_dir=${custom_out_dir} \
  --tree_subsample_frac=${tree_subsample_frac} \
  --tree_subsample_order=${tree_subsample_order} \

