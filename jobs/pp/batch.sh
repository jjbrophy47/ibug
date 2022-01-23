run='jobs/pp/runner.sh'
o='jobs/logs/pp/'
t='lgb'
d=1

fold_list=(1 2 3 4 5)
g_list=(0 1)

for f in ${fold_list[@]}; do
    sbatch -a 1-22             -c 10 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $d 0
    sbatch -a 1-10,12-19,21-22 -c 10 -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d 0
    sbatch -a 11,20            -c 10 -t 2880 -p 'long'  -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d 0
    for g in ${g_list[@]}; do
        sbatch -a 1-22             -c 10 -t 1440 -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $d $g
        sbatch -a 1-10,12-19,21-22 -c 10 -t 1440 -p 'short' -o ${o}'kgbm-%a.out'     $run $f 'kgbm'     $t $d $g
        sbatch -a 11,20            -c 10 -t 2880 -p 'long'  -o ${o}'kgbm-%a.out'     $run $f 'kgbm'     $t $d $g
        sbatch -a 1-10,12-19,21-22 -c 10 -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g
        sbatch -a 11,20            -c 10 -t 2880 -p 'long'  -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g
    done
done

# scratch pad

t_list=('lgb' 'xgb' 'cb')
g_list=(0 1)

for f in ${fold_list[@]}; do
    sbatch -a 1-22             -c 10 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $d 0
done

for f in ${fold_list[@]}; do
    sbatch -a 11            -c 15 -t 10080 -p 'long'  -o ${o}'pgbm-%a.out' $run $f 'pgbm' $t $d 1
done
