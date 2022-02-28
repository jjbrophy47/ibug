run='jobs/predict/runner.sh'
o='jobs/logs/predict/'
t='lgb'
s='crps'
p='short'
td=0
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
tree_list=('lgb' 'xgb' 'cb')

for f in ${fold_list[@]}; do
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t 'nll' 'crps' 1
done

for f in ${fold_list[@]}; do
    for tree in ${tree_list[@]}; do
        sbatch -a 1-10,12-22 -c 4  -t 1440 -p 'short' -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
        sbatch -a 11         -c 10 -t 1440 -p 'short' -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
    done
done

# special case: XGB
for f in ${fold_list[@]}; do
    sbatch -a 20 -c 4 -t 2880 -p 'long' -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
done


# Tree subsampling
tree_subsample_order_list=('random' 'ascending' 'descending')
tree_subsample_frac_list=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for i in ${!tree_subsample_order_list[@]}; do

    to=${tree_subsample_order_list[$i]}
    cd='tree_subsample_'${to}

    for tf in ${tree_subsample_frac[@]}; do
        for f in ${fold_list[@]}; do
            for tree in ${tree_list[@]}; do
                sbatch -a 1-10,12-22 -c 4  -t 1440 -p $p -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $cd $tf $to
                sbatch -a 11         -c 10 -t 1440 -p $p -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $cd $tf $to
            done
        done
    done
done


# Posterior modeling
custom_dir_list=('dist' 'dist_fl' 'dist_fls')

for cd in ${custom_dir_list[@]}; do
    for f in ${fold_list[@]}; do
        for tree in ${tree_list[@]}; do
            sbatch -a 1-10,12-22 -c 4  -t 1440 -p $p -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $cd $tf $to
            sbatch -a 11         -c 10 -t 1440 -p $p -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $cd $tf $to
        done
    done
done
