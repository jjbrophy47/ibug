run='jobs/train/runner.sh'
o='jobs/logs/train/'
t='lgb'
g=1
s='crps'
cd='default'
fold_list=(1 2 3 4 5 6 7 8 9 10)
tree_list=('lgb' 'xgb' 'cb')

# handles most cases
for f in ${fold_list[@]}; do
    sbatch -a 1-22                   -c 4  -t 1440 -p 'preempt' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $g $s
    sbatch -a 1-22                   -c 4  -t 1440 -p 'preempt' -o ${o}'cbu-%a.out'      $run $f 'cbu'      $t $g $s
    sbatch -a 2-6,8-9,12,15-17,21,22 -c 10 -t 1440 -p 'preempt' -o ${o}'bart-%a.out'     $run $f 'bart'     $t $g $s
done

# specific datasets/methods that need more resources
for f in ${fold_list[@]}; do
    sbatch -a 2-6,8,9,12,15-17,21,22 -c 7  -t 1440  -p 'preempt' -o ${o}'pgbm-%a.out'   $run $f 'pgbm'   $t $g $s
    sbatch -a 1,19                   -c 7  -t 2880  -p 'preempt' -o ${o}'pgbm-%a.out'   $run $f 'pgbm'   $t $g $s
    sbatch -a 10,14,18               -c 7  -t 7200  -p 'preempt' -o ${o}'pgbm-%a.out'   $run $f 'pgbm'   $t $g $s
    sbatch -a 7,13,20                -c 15 -t 7200  -p 'preempt' -o ${o}'pgbm-%a.out'   $run $f 'pgbm'   $t $g $s
    sbatch -a 11                     -c 20 -t 7200  -p 'preempt' -o ${o}'pgbm-%a.out'   $run $f 'pgbm'   $t $g $s
done

# IBUG tree variants
tree_list=('xgb' 'cb')

for f in ${fold_list[@]}; do
    for t in ${tree_list[@]}; do
        # sbatch -a 1-10,12-19,21-22 -c 4  -t 1440  -p 'preempt' -o ${o}'ibug-%a.out' $run $f 'ibug' $t $g $s
        sbatch -a 11,20            -c 10 -t 1440  -p 'preempt' -o ${o}'ibug-%a.out' $run $f 'ibug' $t $g $s
    done
done

# IBUG conditional mean variants
cond_mean_type_list=('base' 'neighbors')

for f in ${fold_list[@]}; do
    for cmt in ${cond_mean_type_list[@]}; do
        sbatch -a 1-10,12-19,21-22 -c 4  -t 1440  -p 'preempt' -o ${o}'ibug-%a.out' $run $f 'ibug' 'cb' $g $s $cd $cmt
        sbatch -a 11,20            -c 10 -t 1440  -p 'preempt' -o ${o}'ibug-%a.out' $run $f 'ibug' 'cb' $g $s $cd $cmt
    done
done

# kNN variants
tree_list=('knn' 'lgb' 'cb')
cond_mean_type_list=('base' 'neighbors')

for t in ${tree_list[@]}; do
    for cmt in ${cond_mean_type_list[@]}; do
        for f in ${fold_list[@]}; do
            sbatch -a 1-10,12-19,21-22 -c 4  -t 1440 -p 'preempt' -o ${o}'knn-%a.out' $run $f 'knn' $t $g $s $cd $cmt
            sbatch -a 11,20            -c 10 -t 2880 -p 'preempt' -o ${o}'knn-%a.out' $run $f 'knn' $t $g $s $cd $cmt
        done
    done
done

# scratch pad
fold_list=(1 2 3 4 5 6 7 8 9 10)
for f in ${fold_list[@]}; do
    # sbatch -a 1-10,12-19,21-22 -c 4 -t 1440 -p 'short' -o ${o}'knn-%a.out' $run $f 'knn' 'cb' $g 'nll' 'default' 'neighbors'
    sbatch -a 11 -c 12 -t 1440 -p 'short' -o ${o}'knn-%a.out' $run $f 'knn' 'cb' $g 'nll' 'default' 'neighbors'
    # sbatch -a 1-10,12-19,21-22 -c 12 -t 1440 -p 'short' -o ${o}'ibug-%a.out' $run $f 'ibug' 'xgb' $g 'nll' 'default' 'base'
    # sbatch -a 20 -c 12 -t 1440 -p 'short' -o ${o}'ibug-%a.out' $run $f 'ibug' 'xgb' $g 'nll' 'default' 'base'
    # sbatch -a 1-22                   -c 4  -t 1440  -p 'preempt'  -o ${o}'knn-%a.out'    $run $f 'knn'    $t $g $s
    # sbatch -a 1-10,12-19,21-22       -c 4  -t 1440  -p 'short'  -o ${o}'knn_fi-%a.out' $run $f 'knn_fi' $t $g $s
    # sbatch -a 20,11                  -c 5  -t 2880  -p 'long'  -o ${o}'knn_fi-%a.out' $run $f 'knn_fi' $t $g $s
    # sbatch -a 2 -c 28 -t 1440 -p 'preempt' -o ${o}'bart-%a.out'     $run $f 'bart'     $t $g $s
    # sbatch -a 11 -c 5  -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $g 'crps'
    # sbatch -a 21 -c 4  -t 1440 -p 'preempt' -o ${o}'cbu-%a.out'      $run $f 'cbu'      $t $g $s
done
