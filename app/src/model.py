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

        if self.selector == 'kNN':

            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.model_selection import GridSearchCV, KFold
            from evaluation_utils import r2_pearson

            print('-> 5 folds cross validation and grid searching...')

            inner_cv = KFold(5, shuffle=True, random_state=1673)

            parameters = {'n_neighbors': range(1, 20)}

            model = KNeighborsRegressor(weights='distance')
            clf = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2_pearson)

            self.model = clf
            # ------------------------------------

            from sklearn.model_selection import cross_val_score, KFold
            from evaluation_utils import MAPE, r2_pearson

            # evaluate ---------------------------
            outer_cv = KFold(5, shuffle=True, random_state=75788)
            score = cross_val_score(self.model, X, y, scoring=r2_pearson, cv=outer_cv)
            score_MAPE = cross_val_score(self.model, X, y, scoring=MAPE, cv=outer_cv)

            print('-> scores: ', score)
            # score
            results = {
                'score': score.mean(),
                'MAPE': score_MAPE.mean(),
                'time': str(datetime.datetime.now())
            }

            with open('../app/logs/results.txt', 'w') as file:
                file.write(json.dumps(results))

            self.model.fit(X, y)

            print('-> scores written to disk. Best parameters:', self.model.best_params_)


