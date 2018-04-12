class Modeller:
    """
    class to handles the ensambling of models that use satellite and other features to predict indicator.
    """
    def __init__(self, model_list, sat_features=None):

        self.model_list = model_list
        self.scores = {}
        self.vars = {}
        self.kNN = None
        self.Kriging = None
        self.RmSense = None
        self.sat_features = sat_features

    def compute(self, X, y):
        from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
        from evaluation_utils import r2

        inner_cv = KFold(5, shuffle=True, random_state=1673)
        outer_cv = KFold(5, shuffle=True, random_state=75788)
        print('-> cross validation and grid searching...')

        if 'kNN' in self.model_list:
            from sklearn.neighbors import KNeighborsRegressor

            parameters = {'n_neighbors': range(1, 20, 2)}
            model = KNeighborsRegressor(weights='distance')
            self.kNN = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)
            score = cross_val_score(self.kNN, X, y, scoring=r2, cv=outer_cv)
            self.scores['kNN'], self.vars['kNN'] = score.mean(), score.var()
            print('INFO: kNN score ', score.mean())

        if 'Kriging' in self.model_list:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF

            parameters = {"kernel": [RBF(l) for l in [[1, 1]]]}
            model = GaussianProcessRegressor(alpha=0.1, n_restarts_optimizer=0)
            self.Kriging = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)
            score = cross_val_score(self.Kriging, X, y, scoring=r2, cv=outer_cv)
            self.scores['Kriging'], self.vars['Kriging'] = score.mean(), score.var()
            print('INFO: Kriging score ', score.mean())

        if 'RmSense' in self.model_list:
            from sklearn.linear_model import Ridge

            model = Ridge()
            self.RmSense = GridSearchCV(estimator=model, param_grid={"alpha":[0.001,0.01,0.1,1,10,100,100]},
                                        cv=inner_cv, scoring=r2)
            score = cross_val_score(self.RmSense, self.sat_features, y, scoring=r2, cv=outer_cv)
            self.scores['RmSense'], self.vars['RmSense'] = score.mean(), score.var()
            print('INFO: remote sensing score ', score.mean())

        # combine kNN and RmSense
        prd_int = self.kNN.fit(X, y).predict(X)
        prd_rms = self.RmSense.fit(self.sat_features, y).predict(self.sat_features)

        from evaluation_utils import R2
        self.scores['combined'] = R2(y, (prd_int+prd_rms)/2)
        print('score combined: ', self.scores['combined'].mean())


    def save_models(self, id):
        from sklearn.externals import joblib
        import os

        if not os.path.exists('../Models'):
            os.makedirs('../Models')

        for model in self.model_list:

            joblib.dump(exec("self." + model), '../Models/{}_model_config_id_{}.pkl'.format(model, id))

