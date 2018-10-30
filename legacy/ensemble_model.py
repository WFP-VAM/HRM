import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from mlxtend.regressor import StackingRegressor
from sklearn.svm import SVR
from mlxtend.feature_selection import ColumnSelector

class MeanRegressor(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return(np.mean(X, axis=1))

class MyEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Base class for all ensemble classes.

    This estimator ensemle different estimators each one with it's own input set

    Parameters
    ----------
    regressors : array-like, shape = [n_regressors]
        A list of regressors.
        Invoking the `fit` method on the `MyEnsembleRegressor` will fit clones
        of those original regressors that will
        be stored in the class attribute
        `self.regr_`.
    meta_regressor : object, default SVR(kernel='rbf')
        The meta-regressor to be fitted on the ensemble of
        regressors

    Examples
    --------
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from Src.ensemble_model import MyEnsembleRegressor


    bunch = load_boston()
    X, y = load_boston(return_X_y=True)
    df = pd.DataFrame(data=bunch.data, columns=bunch.feature_names)
    df['target'] = y

    input_model1 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM']
    input_model2 = ['AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'target']

    # Define your regressor
    rndf = RandomForestRegressor(n_estimators=10)
    lr = LinearRegression()

    # Define the ensemble regressor
    mymodel = MyEnsembleRegressor(regressors=[rndf, lr], meta_regressor=None)


    # You can fit all your estimator
    mymodel.fit(df, df['target'])
    mymodel.predict(df)
    mymodel.score(df, df['target'])

    # You can select different input features for each model
    mymodel.fit(df, df['target'], column_selection=[input_model1, input_model2])
    mymodel.predict(df)
    mymodel.score(df, df['target'])
    """

    def __init__(self, regressors, meta_regressor=None):
        self.my_esitimators = regressors
        self.meta_regressor = meta_regressor

    def _make_estimators(self):
        """
        Create a stacking regessor made of pipelines.
        The pipelines are contains the column selector transformer

        """
        pipes = list()
        if self.columns_selection is None:
            for estimator in self.my_esitimators:
                pipes.append(make_pipeline(estimator))
        else:
            for idx, estimator in enumerate(self.my_esitimators):
                pipes.append(make_pipeline(ColumnSelector(cols=self.columns_selection[idx]), estimator))

        if self.meta_regressor is None:
            self.meta_regressor = MeanRegressor()
            
        stregr = StackingRegressor(regressors=pipes, meta_regressor=self.meta_regressor)

        return stregr

    def fit(self, X, y, column_selection=None):
        """
        Fit the model
        Parameters
        ----------
        X : pandas.DataFrame or {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : pandas.Series or array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values.
        column_selection : list of list
            Selected columns for each regressor.
            If X is a DataFrame, column_selection elements should be a list of column name.

        Returns
        -------

        """
        self.columns_selection = self._check_column_selection(X, columns_selction=column_selection)
        self.ensembled_esitimator_ = self._make_estimators()
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
        """ Predict target values for X.

        Parameters
        ----------
        X : pd.DataFrame or {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        y_target : array-like, shape = [n_samples] or [n_samples, n_targets]
            Predicted target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.as_matrix()
        return self.ensembled_esitimator_.predict(X)



