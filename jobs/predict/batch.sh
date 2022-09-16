run='jobs/predict/runner.sh'
o='jobs/logs/predict/'
t='lgb'
s='crps'
td=0
ci='default'
co='default'
tf=1.0
to='random'
nj=-1
fold_list=(1 2 3 4 5 6 7 8 9 10)

# baselines
for f in ${fold_list[@]}; do
    # sbatch -a 1-22                   -c 4  -t 1440 -p 'preempt' -o ${o}'ngboost-%a.out'  $run $f 'ngboost' $t $s $s $td

    sbatch -a 1-22 -c 4  -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'    $t 'crps' $s 1
    # sbatch -a 1-22                   -c 4  -t 1440 -p 'preempt' -o ${o}'cbu-%a.out'      $run $f 'cbu'     $t 'nll' $s 1
    # sbatch -a 2-6,8-9,12,15-17,21,22 -c 10 -t 1440 -p 'preempt' -o ${o}'bart-%a.out'     $run $f 'bart'    $t 'nll' $s 1
done

# IBUG tree variants
tree_list=('xgb')

for f in ${fold_list[@]}; do
    for tree in ${tree_list[@]}; do
        sbatch -a 1-10,12-19,21-22 -c 4  -t 1440 -p 'preempt'  -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
        sbatch -a 11,20            -c 10 -t 2880 -p 'preempt'  -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td
    done
done

# IBUG conditional mean variants
fold_list=(1 2 3 4 5 6 7 8 9 10)
cond_mean_type_list=('base' 'neighbors')

for f in ${fold_list[@]}; do
    for cmt in ${cond_mean_type_list[@]}; do
        sbatch -a 1-10,12-19,21-22 -c 4  -t 1440 -p 'preempt'  -o ${o}'ibug-%a.out' $run $f 'ibug' 'lgb' $s $s $td $ci $co $cmt
        sbatch -a 11,20            -c 10 -t 2880 -p 'preempt'  -o ${o}'ibug-%a.out' $run $f 'ibug' 'lgb' $s $s $td $ci $co $cmt
    done
done

# kNN variants
tree_list=('cb')
cond_mean_type_list=('base' 'neighbors')

for t in ${tree_list[@]}; do
    for cmt in ${cond_mean_type_list[@]}; do
        for f in ${fold_list[@]}; do
            sbatch -a 1-10,12-19,21-22 -c 4  -t 1440 -p 'short' -o ${o}'knn-%a.out' $run $f 'knn' $t $s $s $td $ci $co $cmt
            sbatch -a 11,20            -c 10 -t 2880 -p 'long'  -o ${o}'knn-%a.out' $run $f 'knn' $t $s $s $td $ci $co $cmt
        done
    done
done

# scratch pad
fold_list=(1 2 3 4 5 6 7 8 9 10)
for f in ${fold_list[@]}; do
    sbatch -a 11 -c 7 -t 1440 -p 'short' -o ${o}'knn-%a.out' $run $f 'knn' 'cb' 'nll' 'nll' $td $ci $co 'base'
done

# NGBoost and PGBM as base models
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
tree_list=('ngboost' 'pgbm')
for f in ${fold_list[@]}; do
    for tree in ${tree_list[@]}; do
        sbatch -a 1-22 -c 7 -t 1440 -p $p -o ${o}'ibug-%a.out' $run $f 'ibug' $tree $s $s $td $ci $co $tf $to $nj
    done
done


# Tree subsampling
tree_list=('cb')  # cb lgb
tree_subsample_order_list=('descending')  # random ascending descending
tree_subsample_frac_list=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
op=${o}'ibug-%a.out'

for t in ${tree_list[@]}; do
    for i in ${!tree_subsample_order_list[@]}; do
        to=${tree_subsample_order_list[$i]}
        cd='tree_subsample_'${to}
        for tf in ${tree_subsample_frac_list[@]}; do
            for f in ${fold_list[@]}; do
                sbatch -a 1-10,12-19,21-22 -c 4 -t 600  -p 'short' -o $op $run $f 'ibug' $t $s $s $td $ci $cd 'base' $tf $to
                # sbatch -a 11,20         -c 7 -t 2880 -p 'long'  -o $op $run $f 'ibug' $t $s $s $td $ci $cd 'base' $tf $to
            done
        done
    done
done


# Posterior modeling
tree_list=('lgb')
custom_dir_list=('dist')  # dist_fl, dist_fls
op=${o}'ibug-%a.out'

for t in ${tree_list[@]}; do
    for cd in ${custom_dir_list[@]}; do
        for f in ${fold_list[@]}; do
            sbatch -a 1-22 -c 10 -t 1440 -p 'short' -o $op $run $f 'ibug' $t $s $s $td $ci $cd 'base' $tf $to 10
            # sbatch -a 1-10,12-19,21-22 -c 4  -t 1440 -p 'short' -o $op $run $f 'ibug' $t $s $s $td $ci $cd 'base' $tf $to 4
            # sbatch -a 20               -c 4  -t 2880 -p 'long'  -o $op $run $f 'ibug' $t $s $s $td $ci $cd 'base' $tf $to 4
            # sbatch -a 11               -c 10 -t 4320 -p 'long'  -o $op $run $f 'ibug' $t $s $s $td $ci $cd 'base' $tf $to 10
        done
    done
done


# CBU+IBUG
tree_type_list=('lgb' 'cb')
metric_list=('crps' 'nll')
for t in ${tree_type_list[@]}; do
    for metric in ${metric_list[@]}; do
        sbatch -c 4 -t 1440 -p 'short' -o ${o}'cbu_ibug.out' 'scripts/postprocess/cbu_ibug_predict.sh' $t $metric
    done
done
