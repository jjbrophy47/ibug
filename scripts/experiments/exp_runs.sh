#!/bin/bash

model=$1
dataset=$2
delta=$3

dataset_list=('ames_housing' 'cal_housing' 'concrete' 'energy' 'heart' 'kin8nm'
              'life' 'naval' 'obesity' 'online_news' 'power' 'protein'
              'synth_regression' 'wine' 'yacht')
dataset_list=('naval' 'obesity' 'online_news' 'power' 'protein'
              'synth_regression' 'wine' 'yacht')
fold_list=(1 2 3 4 5)

if [[ $dataset = 'list' ]]; then
    for dataset in ${dataset_list[@]}; do
        for fold in ${fold_list[@]}; do
            python3 scripts/experiments/prediction.py --dataset $dataset --model $model --delta $delta --fold $fold
        done
    done
else
    for fold in ${fold_list[@]}; do
        python3 scripts/experiments/prediction.py --dataset $dataset --model $model --delta $delta --fold $fold
    done
fi
