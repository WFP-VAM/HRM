import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone


class MyEnsemble(BaseEstimator, MetaEstimatorMixin):
    """Base class for all ensemble classes.

    This estimator ensemle different estimators each one with it's own input set

    Parameters
    ----------
    base_estimators : dict of object, optional (default=None)
        The base estimator from which the ensemble is built.
        es: {'model1': RandomForest, 'model2':DecisionTree}

    estimator_params : dict of dict of param names and param values
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.
        es: {'model1': {'n_estimators':10}, 'model2':{'max_depth':5}}
    """

    def __init__(self, base_estimators, estimator_params=None):
        self.my_esitimators = base_estimators
        self.my_estimator_params = estimator_params

        self.my_esitimators_ = self._make_estimators()

    def _make_estimators(self):
        """
        Make and configure a copy of the `base_estimator_` attribute.

        """
        my_esitimators_ = dict()
        for k, v in self.my_esitimators.items():
            my_esitimators_[k] = v(**self.my_estimator_params[k])

        return my_esitimators_

    def fit(self, X, y):
        """

        Parameters
        ----------
        X: Dict of array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.
            es: {'model1': df[x_values], 'model2': df[x_values2]}
        y: Series or array
            Supervised output
        Returns
        -------
        self : object
            Returns self.
        """

        for k, v in self.my_esitimators_.items():
            v.fit(X[k], y)

    def predict(self, X):

        y = None
        for k, v in self.my_esitimators_.items():
            y_ = v.predict(X[k]).reshape(-1,1)
            if y is None:
                y = y_
            else:
                y = np.concatenate((y, y_), 1)

        return np.mean(y, axis=1)


