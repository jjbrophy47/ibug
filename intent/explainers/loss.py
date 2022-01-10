from .base import Explainer
from .parsers import util


class Loss(Explainer):
    """
    Explainer that randomly returns higher influence
        for train examples with larger loss.

    Global-Influence Semantics
        - More positive values are assigned to train examples with higher loss.

    Note
        - Supports GBDTs and RFs.
    """
    def __init__(self, logger=None):
        """
        Input
            logger: object, If not None, output to logger.
        """
        self.logger = logger

    def fit(self, model, X, y):
        """
        Input
            model: tree ensemble.
            X: 2d array of train examples.
            y: 1d array of train targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        self.original_model_ = model

        self.n_class_ = self.model_.n_class_
        self.objective_ = self.model_.objective
        self.loss_fn_ = util.get_loss_fn(self.objective_, self.n_class_, self.model_.factor)

        return self

    def get_self_influence(self, X, y, batch_size=None):
        """
        Input
            X: 2d array of test data.
            y: 2d array of test targets.
            batch_size: Unused, exists for compatibility.

        Return
            - 1d array of shape=(no. train,).
                * Arrays are returned in the same order as the traing data.
        """
        return self._get_loss(self.loss_fn_, self.original_model_, self.objective_,
                              self.X_train_, self.y_train_)

    def get_local_influence(self, X, y):
        """
        - Compute influence of each training instance on the test loss.

        Input
            X: 2d array of test examples.
            y: 1d array of test targets.
                * Could be the actual label or the predicted label depending on the explainer.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        raise ValueError('get_local_influence not implemented for Loss explainer.')

    # private
    def _get_loss(self, loss_fn, model, objective, X, y, batch=False):
        """
        Return
            - 1d array of individual losses of shape=(X.shape[0],),
                unless batch=True, then return a single float.

        Note
            - Parallelizable method.
        """
        if objective == 'regression':
            y_pred = model.predict(X)  # shape=(X.shape[0])

        elif objective == 'binary':
            y_pred = model.predict_proba(X)[:, 1]  # 1d arry of pos. probabilities

        else:
            assert objective == 'multiclass'
            y_pred = model.predict_proba(X)  # shape=(X.shape[0], no. class)

        result = loss_fn(y, y_pred, raw=False, batch=batch)  # shape(X.shape[0],) or single float

        return result
