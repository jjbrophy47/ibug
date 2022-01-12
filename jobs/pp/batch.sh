run='jobs/pp/runner.sh'
o='jobs/logs/pp/'
t='lgb'
b='none'

fold_list=(1 2 3 4 5)

for f in ${fold_list[@]}; do
    sbatch -a 1-15     -c 10 -t 1440 -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $b
    sbatch -a 1-15     -c 10 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $b
    sbatch -a 1-6,8-15 -c 10 -t 1440 -p 'short' -o ${o}'kgbm-%a.out'     $run $f 'kgbm'     $t $b
    sbatch -a 1-6,8-15 -c 10 -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $b
    sbatch -a 1-6,8-15 -c 10 -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $b
done

for f in ${fold_list[@]}; do
    sbatch -a 7 -c 10 -t 2880 -p 'long' -o ${o}'kgbm-%a.out'     $run $f 'kgbm'     $t $b
    sbatch -a 7 -c 10 -t 2880 -p 'long' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $b
    # sbatch -a 7 -c 10 -t 2880 -p 'long' -o ${o}'pgbm-%a.out'     $run $f 'pgbm'     $t $b
done
