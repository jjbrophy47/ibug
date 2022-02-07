run='jobs/pp2/runner.sh'
o='jobs/logs/pp2/'
t='lgb'
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
g_list=(1)
d_list=(1)
tree_list=('lgb' 'xgb' 'cb')

for f in ${fold_list[@]}; do
    for d in ${d_list[@]}; do
        sbatch -a 1-22             -c 4 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $d 0
        sbatch -a 1-22             -c 4 -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d 0
    done
done

for f in ${fold_list[@]}; do
    for d in ${d_list[@]}; do
        for g in ${g_list[@]}; do
            sbatch -a 1-22             -c 4  -t 1440  -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $d $g
            sbatch -a 1-10,12-19,21-22 -c 7  -t 1440  -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g
            sbatch -a 11,20            -c 15 -t 7200  -p 'long'  -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g
        done
    done
done

for f in ${fold_list[@]}; do
    for d in ${d_list[@]}; do
        for g in ${g_list[@]}; do
            for tree in ${tree_list[@]}; do
                sbatch -a 1-22 -c 4 -t 1440  -p 'short' -o ${o}'kgbm-%a.out' $run $f 'kgbm' $tree $d $g
            done
        done
    done
done
