IBUG: Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees
---

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
