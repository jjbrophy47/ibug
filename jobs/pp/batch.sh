run='jobs/pp/runner.sh'
o='jobs/logs/pp/'
t='lgb'
b='none'

sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'constant-%a.out' $run 'constant' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run 'knn' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'kgbm-%a.out'     $run 'kgbm' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run 'ngboost' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'pgbm-%a.out'     $run 'pgbm' $t $b
