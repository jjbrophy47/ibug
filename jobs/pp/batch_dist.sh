run='jobs/pp/runner.sh'
o='jobs/logs/dist/'
t='lgb'
d=1
g=1
custom_dir_list=('dist' 'fl_dist' 'fls_dist')
fold_list=(1 2 3 4 5)

for f in ${fold_list[@]}; do
    for c in ${custom_dir_list[@]}; do
        sbatch -a 1-22 -c 10 -t 1440  -p 'short' -o ${o}'kgbm-%a.out' $run $f 'kgbm' $t $d $g $c
    done
done
