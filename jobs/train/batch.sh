run='jobs/train/runner.sh'
o='jobs/logs/train/'
t='lgb'
g=1
s='nll'
fold_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
tree_list=('lgb' 'xgb' 'cb')

for f in ${fold_list[@]}; do
    sbatch -a 1-22             -c 4 -t 1440 -p 'short' -o ${o}'knn-%a.out'      $run $f 'knn'      $t $g $s
    sbatch -a 1-22             -c 4 -t 1440 -p 'short' -o ${o}'ngboost-%a.out'  $run $f 'ngboost'  $t $g $s
    sbatch -a 1-22             -c 4 -t 1440 -p 'short' -o ${o}'constant-%a.out' $run $f 'constant' $t $g $s
done

for f in ${fold_list[@]}; do
    sbatch -a 1-6,8-10,12,14-19,21-22 -c 7  -t 1440  -p 'short' -o ${o}'pgbm-%a.out' $run $f 'pgbm' $t $g $s
    sbatch -a 7,11,13,20              -c 15 -t 7200  -p 'long'  -o ${o}'pgbm-%a.out' $run $f 'pgbm' $t $g $s
done

for f in ${fold_list[@]}; do
    for tree in ${tree_list[@]}; do
        sbatch -a 1-22 -c 4 -t 1440  -p 'short' -o ${o}'ibug-%a.out' $run $f 'ibug' $t $t $g $s
    done
done