run='jobs/predict/runner.sh'
o='jobs/logs/predict/'
t='lgb'
s='nll'
p='short'
l='long'
td=0
co='default'
tf=1.0
to='random'
nj=-1
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
tree_list=('lgb' 'xgb' 'cb')

for f in ${fold_list[@]}; do
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t 'nll' 'crps' 1
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'knn_fi-%a.out'   $run $f 'knn_fi'   $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'cbu-%a.out'      $run $f 'cbu'      $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'cbu-%a.out'      $run $f 'cbu'      $t 'nll' 'crps' 1
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'bart-%a.out'     $run $f 'bart'     $t $s $s $td
    sbatch -a 1-22 -c 4 -t 1440 -p 'short' -o ${o}'bart-%a.out'     $run $f 'bart'     $t 'nll' 'crps' 1
done

for f in ${fold_list[@]}; do
    for tree in ${tree_list[@]}; do
        sbatch -a 1-10,12-19,21-22 -c 4  -t 1440 -p 'short' -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
        sbatch -a 20               -c 4  -t 2880 -p 'long'  -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
        sbatch -a 11               -c 7  -t 2880 -p 'long'  -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
    done
done

# scratch pad
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
tree_list=('ngboost' 'pgbm')
for f in ${fold_list[@]}; do
    for tree in ${tree_list[@]}; do
        sbatch -a 1-22 -c 7 -t 1440 -p $p -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $co $tf $to $nj
    done
done


# Tree subsampling
tree_subsample_order_list=('random' 'ascending' 'descending')
tree_subsample_frac_list=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
op=${o}'ibug-%a.out'

for tree in ${tree_list[@]}; do
    for i in ${!tree_subsample_order_list[@]}; do
        to=${tree_subsample_order_list[$i]}
        cd='tree_subsample_'${to}
        for tf in ${tree_subsample_frac_list[@]}; do
            for f in ${fold_list[@]}; do
                # sbatch -a 1-10,11-19,21-22 -c 4  -t 300 -p $p -o $op $run $f 'ibug' $tree $s $s $td $cd $tf $to
                sbatch -a 20 -c 4 -t 2880 -p 'long' -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
                sbatch -a 11 -c 7 -t 2880 -p 'long' -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
            done
        done
    done
done


# Posterior modeling
custom_dir_list=('dist' 'dist_fl' 'dist_fls')

for cd in ${custom_dir_list[@]}; do
    for f in ${fold_list[@]}; do
        for tree in ${tree_list[@]}; do
            sbatch -a 1-10,12-19,21-22 -c 4  -t 1440 -p $p -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $cd
            # sbatch -a 20               -c 4  -t 2880 -p $l -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $cd
            # sbatch -a 11               -c 10 -t 4320 -p $l -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $cd
        done
    done
done
