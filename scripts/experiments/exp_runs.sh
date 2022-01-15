#!/bin/bash

model=$1
dataset=$2

dataset_list=('ames_housing' 'cal_housing' 'concrete' 'energy' 'heart' 'kin8nm'
              'life' 'msd' 'naval' 'obesity' 'online_news' 'power' 'protein'
              'synth_regression' 'wine' 'yacht')
delta_list=(0 1)
fold_list=(1 2 3 4 5)

if [[ $dataset = '' ]]; then
    for dataset in ${dataset_list[@]}; do
        for fold in ${fold_list[@]}; do
            python3 scripts/experiments/prediction.py --dataset $dataset --model $model --fold $fold
        done
    done
else
    for fold in ${fold_list[@]}; do
        python3 scripts/experiments/prediction.py --dataset $dataset --model $model --fold $fold
    done
fi
