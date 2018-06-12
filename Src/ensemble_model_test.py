import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from Src.ensemble_model import MyEnsemble


bunch = load_boston()
_, target = load_boston(return_X_y=True)
df = pd.DataFrame(data=bunch.data, columns=bunch.feature_names)
df['target'] = target

input_model1 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM']
input_model2 = ['AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'target']

mymodel = MyEnsemble(base_estimators={'model1': RandomForestRegressor, 'model2': LinearRegression},
                     estimator_params={'model1': {'n_estimators': 5, 'max_depth': 10},
                                       'model2': {'fit_intercept': True}})

x = {'model1': df[input_model1], 'model2': df[input_model2]}
mymodel.fit(X=x, y=df['target'])

y = mymodel.predict(x)



