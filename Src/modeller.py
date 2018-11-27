import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector
from mlxtend.regressor import StackingRegressor


class MeanRegressor(BaseEstimator):
    """Custom Estimator to Average features"""
    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return(np.mean(X, axis=1))


class Modeller:
    """
    Allows to handle rs_features and spatial_features in separate models.
    """
    def __init__(self, X, rs_features=[], spatial_features=["gpsLatitude", "gpsLongitude"], scoring='r2', cv_loops=20):
        """

        :param X: DataFrame, the dataset.
        :param rs_features: list, the remote sensing features as columns of X.
        :param spatial_features: list, the "spatial" indexes, used for spatial models, defautls to ["gpsLatitude", "gpsLongitude"]
        :param scoring: the evaluation function. Defaults to 'r2'.
        :param cv_loops: Bootstrapping loops.
        """
        rs_features = [rs_features] if isinstance(rs_features, str) else rs_features
        self.rs_features_indices = [X.columns.get_loc(c) for c in X.columns if c in rs_features]
        self.spatial_features_indices = (X.columns.get_loc(spatial_features[0]), X.columns.get_loc(spatial_features[1]))
        self.cv_loops = cv_loops
        self.X = X.values
        self.scoring = scoring

    def compute_scores(self, pipeline, y):
        scores = np.array([])
        for i in range(self.cv_loops):
            shuffle = KFold(n_splits=5, shuffle=True, random_state=i)
            scores = np.append(scores, cross_val_score(pipeline, self.X, y, cv=shuffle, scoring=self.scoring))
        return scores

    def make_model_pipeline(self, model):
        """
        creates a scikit-learn pipeline.
        :param model: either "kNN" or "Ridge" only for now.
        :return: scikit-learn pipeline.
        """
        inner_cv = KFold(5, shuffle=True, random_state=1673)
        if model == 'kNN':
            parameters = {'n_neighbors': range(1, 18, 2)}
            estimator = KNeighborsRegressor(weights='distance')
            cols = self.spatial_features_indices
        if model == 'Ridge':
            parameters = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            estimator = Ridge()
            cols = self.rs_features_indices
        gridsearch = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv, scoring=self.scoring, iid=False)
        pipeline = make_pipeline(ColumnSelector(cols=cols), gridsearch)
        return pipeline

    @staticmethod
    def make_ensemble_pipeline(pipelines):
        """
        stacks pipelines together in one model, from mlxtend.
        """
        pipeline = StackingRegressor(regressors=pipelines, meta_regressor=MeanRegressor())
        return pipeline
