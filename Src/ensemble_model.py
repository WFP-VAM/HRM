import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from mlxtend.regressor import StackingRegressor
from sklearn.svm import SVR
from mlxtend.feature_selection import ColumnSelector

class MyEnsembleRegressor(BaseEstimator, RegressorMixin):
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

    def __init__(self, base_estimators, meta_regressor=None):
        self.my_esitimators = base_estimators
        self.meta_regressor = meta_regressor



    def _make_estimators(self, X):
        """
        Make and configure a copy of the `base_estimator_` attribute.

        """
        pipes = list()
        if self.columns_selection is None:
            for estimator in self.my_esitimators:
                pipes.append(make_pipeline(estimator))
        else:
            for idx, estimator in enumerate(self.my_esitimators):
                pipes.append(make_pipeline(ColumnSelector(cols=self.columns_selection[idx]), estimator))

        if self.meta_regressor is None:
            self.meta_regressor = SVR(kernel='rbf')

        stregr = StackingRegressor(regressors=pipes, meta_regressor=self.meta_regressor)

        return stregr

    def fit(self, X, y, column_selection=None):
        self.columns_selection = self._check_column_selection(X, columns_selction=column_selection)
        self.ensembled_esitimator_ = self._make_estimators(X)
        if isinstance(X, pd.DataFrame):
            X = X.as_matrix()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        self.ensembled_esitimator_.fit(X, y)


    def _check_column_selection(self, X, columns_selction):
        if columns_selction is None:
            return None
        if not isinstance(columns_selction, list):
            raise TypeError('columns_selction shold be a list')
        is_list = [isinstance(l, list) for l in columns_selction]
        if is_list.__contains__(False):
            raise TypeError('columns_selction elements should be list of columns')
        if isinstance(X, pd.DataFrame):
            selection_list = list()
            for selection in columns_selction:
                msk = [x in selection for x in X.columns]
                selection_list.append(tuple(np.array(range(0, len(X.columns)))[msk]))
            return selection_list
        else:
            return columns_selction

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.as_matrix()
        return self.ensembled_esitimator_.predict(X)



