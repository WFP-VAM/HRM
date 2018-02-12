from scipy import stats
from sklearn.metrics import make_scorer, r2_score
import numpy as np


# SCORERS
def r2_pearson(ground_truth, predictions):
    r2_pearson = stats.pearsonr(ground_truth, predictions)[0] ** 2
    return r2_pearson


def R2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2


def MAPE(y, yhat):

    diff = np.abs(np.divide((y-yhat), y, out=np.zeros_like(yhat), where=y!=0))

    return(np.sum(diff)/len(y))


r2 = make_scorer(R2, greater_is_better=True) # counter intuitive but otherwise shit results???
r2_pearson = make_scorer(r2_pearson, greater_is_better=True)
MAPE = make_scorer(MAPE, greater_is_better=False)
