run='jobs/pp/runner.sh'
o='jobs/logs/pp/'
t='lgb'
fold_list=(1 2 3 4 5)
g_list=(0 1)
d_list=(0 1)
tree_list=('lgb' 'xgb' 'cb')

for f in ${fold_list[@]}; do

    for d in ${d_list[@]}; do
        sbatch -a 1-22             -c 10 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $d 0
        sbatch -a 1-10,12-19,21-22 -c 10 -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d 0
        sbatch -a 11,20            -c 10 -t 2880 -p 'long'  -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d 0

        for g in ${g_list[@]}; do
            sbatch -a 1-22             -c 10 -t 1440  -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $d $g
            sbatch -a 1-10,12-19,21-22 -c 10 -t 1440  -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g
            sbatch -a 11,20            -c 15 -t 10080 -p 'long'  -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g

            for tree in ${tree_list[@]}; do
                sbatch -a 1-10,12-19,21-22 -c 10 -t 1440  -p 'short' -o ${o}'kgbm-%a.out' $run $f 'kgbm' $tree $d $g
                sbatch -a 11,20            -c 10 -t 7200  -p 'long'  -o ${o}'kgbm-%a.out' $run $f 'kgbm' $tree $d $g
            done
        done
    done
done

# scratch pad

for f in ${fold_list[@]}; do
    for d in ${d_list[@]}; do
        for g in ${g_list[@]}; do
            sbatch -a 11 -c 28 -t 1440  -p 'short'  -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g
        done
    done
done

for f in ${fold_list[@]}; do

    for d in ${d_list[@]}; do
        sbatch -a 11 -c 10 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $d 0
        sbatch -a 11 -c 10 -t 2880 -p 'long'  -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d 0

        for g in ${g_list[@]}; do
            sbatch -a 11 -c 10 -t 1440  -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $d $g
            sbatch -a 11 -c 15 -t 7200  -p 'long'  -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g

            for tree in ${tree_list[@]}; do
                sbatch -a 11 -c 10 -t 4320  -p 'long'  -o ${o}'kgbm-%a.out' $run $f 'kgbm' $tree $d $g
            done
        done
    done
done
