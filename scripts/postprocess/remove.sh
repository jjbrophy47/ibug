#!/bin/bash

in_dir=$1
method=$2

dataset_list=('ames_housing' 'cal_housing' 'concrete' 'energy' 'heart' 'kin8nm'
              'life' 'msd' 'naval' 'obesity' 'online_news' 'power' 'protein'
              'synth_regression' 'wine' 'yacht')
fold_list=(1 2 3 4 5)

for dataset in ${dataset_list[@]}; do
    for fold in ${fold_list[@]}; do
        rm -rf ${in_dir}/${dataset}/fold${fold}/${method}
    done
done
