from evaluation_utils import r2, R2
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge


class Modeller:

    def __init__(self, model_list, sat_features=None):
        """
        class to handles the models that use satellite and other features to predict indicator.
        :param model_list: list of models to use. Available: kNN, Kriging, RmSense
        :param sat_features: if you plan to compute a model on remote sensing features, pass a DataFrame with the features.
        """
        self.model_list = model_list
        self.scores = {}
        self.vars = {}
        self.kNN = None
        self.Kriging = None
        self.RmSense = None
        self.sat_features = sat_features

    def compute(self, X, y):
        """
        Trains all the models listed in self.model_list on X and y. If you compute also a model on
        remote sensing features, it will use self.sat_features for that.
        :param X: DataFrame
        :param y: Series
        """
        inner_cv = KFold(5, shuffle=True, random_state=1673)
        outer_cv = KFold(5, shuffle=True, random_state=75788)
        print('-> cross validation and grid searching...')

        if 'kNN' in self.model_list:

            parameters = {'n_neighbors': range(1, 18, 2)}
            model = KNeighborsRegressor(weights='distance')
            self.kNN = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)
            score = cross_val_score(self.kNN, X, y, scoring=r2, cv=outer_cv)
            self.scores['kNN'], self.vars['kNN'] = score.mean(), score.var()
            print('INFO: kNN score ', score.mean(), score.var())

        if 'Kriging' in self.model_list:

            parameters = {"kernel": [RBF(l) for l in [[1, 1]]]}
            model = GaussianProcessRegressor(alpha=0.1, n_restarts_optimizer=0)
            self.Kriging = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)
            score = cross_val_score(self.Kriging, X, y, scoring=r2, cv=outer_cv)
            self.scores['Kriging'], self.vars['Kriging'] = score.mean(), score.var()
            print('INFO: Kriging score ', score.mean(), score.var())

        if 'RmSense' in self.model_list:

            model = Ridge()
            self.RmSense = GridSearchCV(estimator=model, param_grid={"alpha":[0.001,0.01,0.1,1,10,100,1000]},
                                        cv=inner_cv, scoring=r2)
            score = cross_val_score(self.RmSense, self.sat_features, y, scoring=r2, cv=outer_cv)
            print('INFO: best alpha - ', self.RmSense.fit(X, y).best_params_)
            self.scores['RmSense'], self.vars['RmSense'] = score.mean(), score.var()
            print('INFO: remote sensing score ', score.mean(), score.var())

        # combine kNN and RmSense - custom cross_val
        def _k_fold_cross_validation(X, K):
            for k in range(K):
                training = [x for i, x in enumerate(X) if i % K != k]
                validation = [x for i, x in enumerate(X) if i % K == k]
                yield training, validation, K

        tmp_scores = []
        for training, validation, K in _k_fold_cross_validation(X.index, K=4):

            prd_int = self.kNN.fit(X.loc[training, :], y[training]).predict(X.loc[validation, :])
            prd_rms = self.RmSense.fit(self.sat_features.loc[training, :], y[training]).predict(self.sat_features.loc[validation, :])

            tmp_scores.append(R2(y[validation], (prd_int + prd_rms) / 2))

        self.scores['combined'] = np.array(tmp_scores).mean()
        self.vars['combined'] = np.array(tmp_scores).var()
        print(self.scores['combined'])

        print('score combined: ', self.scores['combined'], self.vars['combined'])

    def save_models(self, id):
        from sklearn.externals import joblib
        import os

        if not os.path.exists('../Models'):
            os.makedirs('../Models')

        for model in self.model_list:

            joblib.dump(exec("self." + model), '../Models/{}_model_config_id_{}.pkl'.format(model, id))


