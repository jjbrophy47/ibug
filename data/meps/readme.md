Medical Expenditure Panel Survey (Regression)
---
* Download the following directory from this [github repository](https://github.com/yromano/cqr/tree/master/get_meps_data) to this directory.
    * `get_meps_data`.
* Run: `cd get_meps_data`.
* Run: `Rscript download_data.R`.
* Replace lines `127-129` in `save_dataset` with the following:
```
priv = df[df[attr].isin(vals)].index
```
```
not_priv = np.setdiff1d(df.index, priv)
```
```
df.loc[priv, attr] = privileged_values[0]
```
```
df.loc[not_priv, attr] = unprivileged_values[0]
```
* Run: `python3 main_clean_and_save_to_csv.py`.
* Run: `cd ..`.
* Run: `python3 preprocess.py`. This creates `data.npy`.
