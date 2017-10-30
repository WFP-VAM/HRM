import os
import yaml
import numpy as np
from sklearn.linear_model import Ridge

with open('../public_config.yml', 'r') as cfgfile:
    public_config = yaml.load(cfgfile)

class Scoring:
    """
    Class
    -----
    Score predictions.

    """

    def __init__(self,hh_data_file,CNN_features_file,indicator):
        from pandas import DataFrame, read_csv
        hh_data = read_csv(hh_data_file)
        CNN_features = read_csv(CNN_features_file)
        data=hh_data.merge(CNN_features,on=["i","j"],how='inner')
        self.X = data[list(set(data.columns) - set(hh_data.columns) -  set(['index','Unnamed: 0', 'index_x', 'index_y']))]
        self.y = data[indicator]

    def r2_pearson(self,ground_truth, predictions):
        from scipy.stats import pearsonr
        r2_pearson = pearsonr(ground_truth, predictions)[0] ** 2
        return r2_pearson

    def MAPE(self,ground_truth, predictions):
        return np.mean(np.abs((ground_truth - predictions) / ground_truth)) * 100

    def PCA_array(self,data_features,n_components=10):
        from sklearn.decomposition import PCA
        from pandas import DataFrame
        #from numpy import transpose
        pca = PCA(n_components)
        pca.fit(data_features.transpose())
        eigenvectors=pca.components_
        PCA_array = transpose(eigenvectors)
        return PCA_array

    def GridSearch(self,model,scorer,n_splits,alphas = [0.1,1,10,100,1000]):
        from sklearn.model_selection import GridSearchCV, KFold
        param = np.array(alphas)
        inner_cv = KFold(n_splits, shuffle=True, random_state=1673)
        clf = GridSearchCV(estimator=model, param_grid=dict(alpha=param), cv=inner_cv, scoring=scorer)
        return clf

    def CrossValScore(self,PCA=False,n_splits=5,model=Ridge(),gridsearch=True,scorer="r2_pearson"):
        """"
        Get the cross validated score of a given model with the different parametrers:
        PCA: Yes or No (with 10 components)
        N_splits: Number of KFolds
        model: Regression model
        gridsearch: look for the best regularization Parameter
        Scorer: explained_variance, neg_mean_absolute_error,neg_mean_squared_error, neg_mean_squared_log_error
        neg_median_absolute_error,r2

        """

        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.metrics import make_scorer
        outer_cv = KFold(n_splits, shuffle=True, random_state=75788)
        if scorer=="r2_pearson":
            scorer=make_scorer(self.r2_pearson)
        if scorer=="MAPE":
            scorer=make_scorer(self.MAPE)
        if gridsearch:
            model=self.GridSearch(model,scorer,n_splits)
        if PCA:
            score = cross_val_score(model, self.PCA_array(self.X), self.y, scoring=scorer,cv=outer_cv)
        else:
            score = cross_val_score(model, self.X, self.y, scoring=scorer,cv=outer_cv)
        print("score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
        return score
