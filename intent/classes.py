from .explainers import BoostIn
from .explainers import BoostInLE
from .explainers import BoostInW1
from .explainers import BoostInW1LE
from .explainers import BoostInW2
from .explainers import BoostInW2LE
from .explainers import Trex
from .explainers import LeafInfluence
from .explainers import LeafInfluenceLE
from .explainers import LeafInfluenceSP
from .explainers import LeafInfluenceSPLE
from .explainers import LeafRefit
from .explainers import LeafRefitLE
from .explainers import LOO
from .explainers import LOOLE
from .explainers import DShap
from .explainers import Random
from .explainers import Minority
from .explainers import Loss
from .explainers import TreeSim
from .explainers import LeafSim
from .explainers import InputSim
from .explainers import Target
from .explainers import SubSample


class TreeExplainer(object):
    """
    Influence-method selector.

    Currently supported models:
        - LGBMRegressor, LGBMClassifier
        - XGBRegressor, XGBClassifier
        - CatBoostRegressor, CatBoostClassifier
        - HistGradientBoostingRegressor, HistGradientBoostingClassifier
        - GradientBoostingRegressor, GradientBoostingClassifier

    Semi-supported models:
        - RandomForestRegressor, RandomForestClassifier

    Currently supported removal explainers:
        - BoostIn (adapted TracIn)
        - BoostInW1 (adapted TracIn w/ leaf weight)
        - BoostInW2 (adapted TracIn, squared leaf weight)
        - TREX (adapted representer-point)
        - LeafInfluenceSP (efficient version of LeafInfluence: single point)
        - LeafInfluence (adapted influence functions)
        - LeafRefit (LOO w/ fixed structure)
        - LeafSim  (similarity based on the weighted-leaf-path tree kernel)
        - TreeSim  (similarity based on an arbitrary tree kernel)
        - InputSim (similarity based on input features)
        - SubSample (Approx. Data Shapley)
        - TMC-Shap (Appox. Data Shapley)
        - LOO (leave-one-out retraining)
        - Target (random from same class as test example)
        - Random

    Currently supported label-estimation explainers:
        - BoostInLE (adapted TracIn w/ label estimation)
        - BoostInLEW1 (adapted TracIn w/ label estimation and leaf weight)
        - BoostInLEW2 (adapted TracIn w/ label estimation and leaf weight, squared)
        - LeafRefitLE (LOO w/ fixed structure and label estimation)
        - LeafInfluenceLE (adapted influence functions w/ label estimation)
        - LeafInfluenceSPLE (efficient LeafInfluence w/ label estimation)
        - LOOLE (leave-one-out retraining w/ label estimation)

    Global-only explainers:
        - Loss (loss of train examples)
        - Minority (random from the minority class)
    """
    def __init__(self, method='boostin', params={}, logger=None):

        if method == 'boostin':
            self.explainer_ = BoostIn(**params, logger=logger)

        elif method == 'boostinW1':
            self.explainer_ = BoostInW1(**params, logger=logger)

        elif method == 'boostinW2':
            self.explainer_ = BoostInW2(**params, logger=logger)

        elif method == 'boostinLE':
            self.explainer_ = BoostInLE(**params, logger=logger)

        elif method == 'boostinW1LE':
            self.explainer_ = BoostInW1LE(**params, logger=logger)

        elif method == 'boostinW2LE':
            self.explainer_ = BoostInW2LE(**params, logger=logger)

        elif method == 'trex':
            self.explainer_ = Trex(**params, logger=logger)

        elif method == 'leaf_inf':
            self.explainer_ = LeafInfluence(**params, logger=logger)

        elif method == 'leaf_infLE':
            self.explainer_ = LeafInfluenceLE(**params, logger=logger)

        elif method == 'leaf_infSP':
            self.explainer_ = LeafInfluenceSP(**params, logger=logger)

        elif method == 'leaf_infSPLE':
            self.explainer_ = LeafInfluenceSPLE(**params, logger=logger)

        elif method == 'leaf_refit':
            self.explainer_ = LeafRefit(**params, logger=logger)

        elif method == 'leaf_refitLE':
            self.explainer_ = LeafRefitLE(**params, logger=logger)

        elif method == 'loo':
            self.explainer_ = LOO(**params, logger=logger)

        elif method == 'looLE':
            self.explainer_ = LOOLE(**params, logger=logger)

        elif method == 'dshap':
            self.explainer_ = DShap(**params, logger=logger)

        elif method == 'random':
            self.explainer_ = Random(**params, logger=logger)

        elif method == 'minority':
            self.explainer_ = Minority(**params, logger=logger)

        elif method == 'loss':
            self.explainer_ = Loss(**params, logger=logger)

        elif method == 'tree_sim':
            self.explainer_ = TreeSim(**params, logger=logger)

        elif method == 'leaf_sim':
            self.explainer_ = LeafSim(**params, logger=logger)

        elif method == 'input_sim':
            self.explainer_ = InputSim(**params, logger=logger)

        elif method == 'target':
            self.explainer_ = Target(**params, logger=logger)

        elif method == 'subsample':
            self.explainer_ = SubSample(**params, logger=logger)

        else:
            raise ValueError(f'Unknown method {method}')

    def fit(self, model, X, y, target_labels=None):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen explainer.

        Input
            model: tree ensemble.
            X: 2d array of train data.
            y: 1d array of train targets.
            target_labels: 1d array of new train targets (LE methods only).

        Return
            Fitted explainer.
        """
        if target_labels is None:
            result = self.explainer_.fit(model, X, y)

        else:
            result = self.explainer_.fit(model, X, y, target_labels=target_labels)

        return result
