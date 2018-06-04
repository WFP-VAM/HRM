from evaluation_utils import r2, R2
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
import collections
import numpy as np


class Modeller:

    def __init__(self, model_list, sat_features=None):
        """
        class to handles the models that use satellite and other features to predict indicator.
        :param model_list: list of models to use. Available: kNN, Kriging, RmSense
        :param sat_features: if you plan to compute a model on remote sensing features, pass a DataFrame with the features.
        """
        self.model_list = model_list
        self.scores = {'kNN': [], 'Kriging': [], 'RmSense': [], 'Ensamble': []}
        self.results = {'kNN': [], 'Kriging': [], 'RmSense': [], 'Ensamble': []}
        self.kNN = None
        self.Kriging = None
        self.RmSense = None
        self.sat_features = sat_features

    @staticmethod
    def _k_fold_cross_validation(X, K, n):
        j = 0
        for i in range(n):
            X = X.sample(frac=1, random_state=i)
            for k in range(K):
                j += 1
                training = [x for i, x in enumerate(X.index) if i % K != k]
                validation = [x for i, x in enumerate(X.index) if i % K == k]
                yield training, validation, j

    def compute(self, X, Y, n):
        """
        Trains all the models listed in self.model_list on X and y. If you compute also a model on
        remote sensing features, it will use self.sat_features for that.
        :param X: DataFrame
        :param y: Series
        """
        inner_cv = KFold(5, shuffle=True, random_state=1673)

        print('-> grid searching and cross validation ...')
        for training, validation, j in self._k_fold_cross_validation(X, 5, n):

            x, y, valid_x, valid_y = X.loc[training, :], Y[training], X.loc[validation, :], Y[validation]
            x_features, valid_features = self.sat_features.loc[training, :], self.sat_features.loc[validation, :]

            if 'kNN' in self.model_list:
                parameters = {'n_neighbors': range(1, 18, 2)}
                model = KNeighborsRegressor(weights='distance')
                self.kNN = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)

                res = self.kNN.fit(x, y).predict(valid_x)
                self.results['kNN'].append(list(res))
                self.scores['kNN'].append(R2(valid_y, res))

            if 'Kriging' in self.model_list:
                parameters = {"kernel": [RBF(l) for l in [[1, 1]]]}
                model = GaussianProcessRegressor(alpha=0.1, n_restarts_optimizer=0)
                self.Kriging = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)

                res = self.Kriging.fit(x, y).predict(valid_x)
                self.results['Kriging'].append(list(res))
                self.scores['Kriging'].append(R2(valid_y, res))

            if 'RmSense' in self.model_list:
                parameters = {"alpha": [1]}
                model = Ridge()
                self.RmSense = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)
                #print('INFO: best alpha - ', self.RmSense.fit(x_features, y).best_params_)

                res = self.RmSense.fit(x_features, y).predict(valid_features)
                self.results['RmSense'].append(list(res))
                self.scores['RmSense'].append(R2(valid_y, res))

            if 'Ensamble' in self.model_list:
                res = (self.RmSense.predict(valid_features) + self.kNN.predict(valid_x)) / 2.
                self.results['Ensamble'].append(list(res))
                self.scores['Ensamble'].append(R2(valid_y, res))

        for m in self.model_list:
            print('score {}: {}'.format(m, np.mean(self.scores[m])))

    def save_models(self, id):
        from sklearn.externals import joblib
        import os

        if not os.path.exists('../Models'):
            os.makedirs('../Models')

        for model in list(set(self.model_list) - set(['Ensamble'])):  # TODO: merge ensamble models into 1 class with fit and predict.
            joblib.dump(eval("self." + model), '../Models/{}_model_config_id_{}.pkl'.format(model, id))
