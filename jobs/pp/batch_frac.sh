run='jobs/pp/runner.sh'
o='jobs/logs/frac/'
t='lgb'
d=1
g=1
c='tree_frac'
tree_frac_list=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
fold_list=(1 2 3 4 5)

for f in ${fold_list[@]}; do
    for tf in ${tree_frac_list[@]}; do
        sbatch -a 1-10,12-19,21-22 -c 10 -t 1440  -p 'short' -o ${o}'kgbm-%a.out' $run $f 'kgbm' $t $d $g $c $tf
        sbatch -a 11,20            -c 10 -t 7200  -p 'long'  -o ${o}'kgbm-%a.out' $run $f 'kgbm' $t $d $g $c $tf
    done
done

# scratch pad
