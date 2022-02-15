run='jobs/pp2/runner.sh'
o='jobs/logs/crps2/'
t='lgb'
g=1
d=1
c='crps'
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for f in ${fold_list[@]}; do
    sbatch -a 1-22             -c 4  -t 500  -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $d $g $c
    sbatch -a 1-22             -c 4  -t 500  -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d $g $c
    sbatch -a 1-10,12-19,21-22 -c 7  -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g $c
    sbatch -a 11,20            -c 15 -t 7200 -p 'long'  -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $d $g $c
    sbatch -a 1-22             -c 4  -t 500  -p 'short' -o ${o}'kgbm-%a.out'     $run $f 'kgbm'     $t $d $g $c
done

for f in ${fold_list[@]}; do
    sbatch -a 1-22             -c 4  -t 500  -p 'preempt' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $d $g $c
done
