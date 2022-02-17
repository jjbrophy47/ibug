run='jobs/pp2/runner.sh'
o='jobs/logs/affinity2/'
t='lgb'
d=1
g=1
custom_dir_list=('affinity_stats')
fold_list=(1)

for f in ${fold_list[@]}; do
    for c in ${custom_dir_list[@]}; do
        sbatch -a 1-22 -c 5 -t 500  -p 'short' -o ${o}'kgbm-%a.out' $run $f 'kgbm' $t $d $g $c
    done
done
