IBUG: Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees
---
[![PyPi version](https://img.shields.io/pypi/v/ibug)](https://pypi.org/project/ibug/)
[![Python version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)](https://pypi.org/project/ibug/)
[![Github License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/jjbrophy47/ibug/blob/master/LICENSE)
[![Build](https://github.com/jjbrophy47/ibug/actions/workflows/wheels.yml/badge.svg?branch=v0.0.4)](https://github.com/jjbrophy47/ibug/actions/workflows/wheels.yml)

**ibug** is a simple wrapper that extends *any* gradient-boosted regression trees (GBRT) model into a probabilistic estimator, and is compatible with all major GBRT frameworks including LightGBM, XGBoost, CatBoost, and SKLearn.

Install
---

```shell
pip install ibug
```

Experiment
---

#### Download a Dataset
1. Follow the instructions in the readme for a dataset in `data/`.

#### Quantify Prediction Uncertainy with LightGBM + IBUG

```
python3 scripts/experiments/train.py --tree_type lgb
python3 scripts/experiments/predict.py --tree_type lgb
```

License
---
[Apache License 2.0](https://github.com/jjbrophy47/ibug/blob/master/LICENSE).

<!--Reference
---
Brophy and Lowd. [Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees](). arXiv 2022.

```
```-->
