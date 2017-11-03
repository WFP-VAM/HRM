from scipy import stats
from sklearn.metrics import make_scorer

# SCORERS
def r2_pearson(ground_truth, predictions):
    r2_pearson=stats.pearsonr(ground_truth, predictions)[0] ** 2
    return r2_pearson

def MAPE(y, yhat):
    diff = np.abs((y-yhat)/y)
    return(np.sum(diff)/len(y))

r2_pearson = make_scorer(r2_pearson, greater_is_better=True)
MAPE  = make_scorer(MAPE, greater_is_better=False)
