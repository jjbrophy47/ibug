# IBUG

**I**nstance-**B**ased **U**ncertainty estimation for **G**radient-boosted regresssion trees is a simple method for turning _any_ gradient-boosted regression trees (GBRT) model into a probabilistic estimator.

### Install

1. Install Python 3.9.6+.
2. Install dependencies and compile source code: `make all`.

### Download a Dataset
1. Follow the instructions in the readme for a dataset in `data/`.

### Run an Experiment
Experiment testing the quality of IBUG's probabilistic predictions on the `yacht` dataset using LightGBM as the base model:

```
python3 scripts/experiments/prdiction.py --dataset yacht --model ibug
```
