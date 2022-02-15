#!/bin/bash

in_dir=$1
method=$2

dataset_list=('ames' 'bike' 'california' 'communities' 'concrete' 'energy'
              'facebook' 'kin8nm' 'life' 'meps' 'msd' 'naval' 'news' 'obesity'
              'power' 'protein' 'star' 'superconductor' 'synthetic' 'wave'
              'wine' 'yacht')
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for dataset in ${dataset_list[@]}; do
    for fold in ${fold_list[@]}; do
        rm -rf ${in_dir}/${dataset}/fold${fold}/${method}
    done
done
