#!/bin/bash
#SBATCH --job-name=CBU+IBUG
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load gcc/7.3
module load miniconda
conda activate jbrophy-20210713

tree_type=$1
metric=$2

if [[ ! $tree_type ]]; then
    metric='lgb'
fi

if [[ ! $metric ]]; then
    metric='crps'
fi

dataset_list=('ames' 'bike' 'california' 'communities' 'concrete' 'energy'
              'facebook' 'kin8nm' 'life' 'meps' 'msd' 'naval' 'news' 'obesity'
              'power' 'protein' 'star' 'superconductor' 'synthetic' 'wave'
              'wine' 'yacht')
fold_list=(1 2 3 4 5 6 7 8 9 10)

for dataset in ${dataset_list[@]}; do
    for fold in ${fold_list[@]}; do
        python3 scripts/postprocess/cbu_ibug_predict.py \
            --dataset=${dataset} \
            --tree_type=${tree_type} \
            --in_scoring=${metric} \
            --out_scoring=${metric} \
            --fold=${fold} \
    done
done
