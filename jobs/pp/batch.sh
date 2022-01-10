run='jobs/pp/runner.sh'
o='jobs/logs/pp/'
t='lgb'
b='none'

sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'_constant-%a.out' $run 'constant' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'_knn-%a.out'      $run 'knn' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'_kgbm-%a.out'     $run 'kgbm' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'_ngboost-%a.out'  $run 'ngboost' $t $b
sbatch -a 1-16  -c 10  -t 1440 -p 'short' -o ${o}'_pgbm-%a.out'     $run 'pgbm' $t $b
