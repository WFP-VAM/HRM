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
    Handles rs_features and spatial_features separately.
    """
    def __init__(self, X, rs_features=None, spatial_features=["gpsLatitude", "gpsLongitude"], scoring='r2', cv_loops=20):
        self.rs_features_indices = [X.columns.get_loc(c) for c in X.columns if c in rs_features]
        self.spatial_features_indices = (X.columns.get_loc(spatial_features[0]), X.columns.get_loc(spatial_features[1]))
        self.cv_loops = cv_loops
        self.X = X.values
        self.scoring = 'r2'

    def compute_scores(self, pipeline, y):
        shuffle = []
        scores = np.array([])
        for i in range(self.cv_loops):
            shuffle = KFold(n_splits=5, shuffle=True, random_state=i)
            scores = np.append(scores, cross_val_score(pipeline, self.X, y, cv=shuffle, scoring=self.scoring))
        return scores

    def make_model_pipeline(self, model):
        inner_cv = KFold(5, shuffle=True, random_state=1673)
        if model == 'kNN':
            parameters = {'n_neighbors': range(1, 18, 2)}
            estimator = KNeighborsRegressor(weights='distance')
            cols = self.spatial_features_indices
        if model == 'Ridge':
            parameters = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            estimator = Ridge()
            cols = self.rs_features_indices
        gridsearch = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv, scoring=self.scoring)
        print(cols)
        pipeline = make_pipeline(ColumnSelector(cols=cols), gridsearch)
        return pipeline

    def make_ensemble_pipeline(self, pipelines):
        pipeline = StackingRegressor(regressors=pipelines, meta_regressor=MeanRegressor())
        return pipeline
