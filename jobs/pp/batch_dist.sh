run='jobs/pp/runner.sh'
o='jobs/logs/dist/'
t='lgb'
d=1
g=1
c='dist'
fold_list=(1 2 3 4 5)

for f in ${fold_list[@]}; do
    sbatch -a 1-10,12-19,21-22 -c 10 -t 1440  -p 'short' -o ${o}'kgbm-%a.out' $run $f 'kgbm' $tree $d $g $c
    sbatch -a 11,20            -c 10 -t 7200  -p 'long'  -o ${o}'kgbm-%a.out' $run $f 'kgbm' $tree $d $g $c
done

# scratch pad
