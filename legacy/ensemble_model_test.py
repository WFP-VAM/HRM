import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from Src.ensemble_model import MyEnsembleRegressor


bunch = load_boston()
X, y = load_boston(return_X_y=True)
df = pd.DataFrame(data=bunch.data, columns=bunch.feature_names)
df['target'] = y

input_model1 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM']
input_model2 = ['AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'target']

# Define your regressor
rndf = RandomForestRegressor(n_estimators=10)
lr = LinearRegression()

# Define the ensemble regressor
mymodel = MyEnsembleRegressor(regressors=[rndf, lr], meta_regressor=None)


# You can fit all your estimator
mymodel.fit(df, df['target'])
mymodel.predict(df)
mymodel.score(df, df['target'])

# You can select different input features for each model
mymodel.fit(df, df['target'], column_selection=[input_model1, input_model2])
mymodel.predict(df)
mymodel.score(df, df['target'])







