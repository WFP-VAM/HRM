import datetime
import json


class IndicatorScaler:
    """
    this class handles the models used to predict the indicator taking as inputs the spatial features.
    """

    def __init__(self, selector, X, y):
        """
        Instantiates the model to use for making predictions.
        selector: str
        X: df
        y: Series
        """
        self.selector = selector
        self.model = None

        from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
        from evaluation_utils import MAPE, r2

        if self.selector == 'kNN':

            from sklearn.neighbors import KNeighborsRegressor

            parameters = {'n_neighbors': range(1, 20)}
            model = KNeighborsRegressor(weights='distance')

        elif self.selector == 'Kriging':

            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF
            parameters = {"kernel": [RBF(l) for l in [[5, 20]]]}
            model = GaussianProcessRegressor(alpha=0.1, n_restarts_optimizer=0)



        inner_cv = KFold(5, shuffle=True, random_state=1673)
        print('-> 5 folds cross validation and grid searching...')

        clf = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2)
        self.model = clf
            # ------------------------------------


        # evaluate ---------------------------
        outer_cv = KFold(5, shuffle=True, random_state=75788)
        score = cross_val_score(self.model, X, y, scoring=r2, cv=outer_cv)
        score_MAPE = cross_val_score(self.model, X, y, scoring=MAPE, cv=outer_cv)

        print('-> scores: ', score)
        # score
        results = {
            'score': score.mean(),
            'MAPE': score_MAPE.mean(),
            'time': str(datetime.datetime.now())
        }

        with open('logs/results.txt', 'w') as file:
            file.write(json.dumps(results))

        self.model.fit(X, y)

        print('-> scores written to disk. Best parameters:', self.model.best_params_)
