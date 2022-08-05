#!/bin/bash

metric=$1

if [[ ! $metric ]]; then
    metric='crps'
fi

dataset_list=('ames' 'bike' 'california' 'communities' 'concrete' 'energy'
              'facebook' 'kin8nm' 'life' 'meps' 'msd' 'naval' 'news' 'obesity'
              'power' 'protein' 'star' 'superconductor' 'synthetic' 'wave'
              'wine' 'yacht')
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for dataset in ${dataset_list[@]}; do
    for fold in ${fold_list[@]}; do
        python3 scripts/postprocess/cbu_ibug_predict.py \
            --dataset=${dataset} \
            --in_scoring=${metric} \
            --out_scoring=${metric} \
            --fold=${fold} \
            --in_dir='output/talapas/experiments/predict/' \
            --out_dir='output/talapas/experiments/predict/'
    done
done
