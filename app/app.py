from flask import Flask, render_template, request
import os
import sys
sys.path.append(os.path.join("..","Src"))
import pandas as pd
from img_lib import RasterGrid
import yaml
import numpy as np
import datetime
import json

app = Flask(__name__, instance_relative_config=True)

with open('../app_config.yml', 'r') as cfgfile:
    config = yaml.load(cfgfile)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/downscale', methods=['POST'])
def master():

    # load dataset -----------------------------------------
    print('-> loading dataset from request...')
    data = pd.read_csv(request.files['file'])

    print('-> loading raster from disk...')
    GRID = RasterGrid(config['raster_file'])
    data['i'], data['j'] = GRID.get_gridcoordinates(data)

    # group stuff, but is ugly

    # predict -----------------------------------------
    X = pd.DataFrame({"i": data["i"], "j": data["j"]})
    y = data.Indicator.values

    # libs
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict
    from evaluation_utils import MAPE, r2_pearson, r2

    print('-> 5 folds cross validation and grid searching...')

    outer_cv = KFold(5, shuffle=True, random_state=75788)
    inner_cv = KFold(5, shuffle=True, random_state=1673)
    print(str(np.datetime64('now')), " INFO: training model ...")

    parameters = {'n_neighbors': range(1, 20)}

    model = KNeighborsRegressor(weights='distance')
    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2_pearson)

    # evaluate
    score = cross_val_score(clf, X, y, scoring=r2_pearson, cv=outer_cv)
    score_MAPE = cross_val_score(clf, X, y, scoring=MAPE, cv=outer_cv)

    # score
    results = {
        'score': score.mean(),
        'MAPE': score_MAPE.mean(),
        'time': str(datetime.datetime.now())
    }
    with open('results.txt', 'w') as file:
        file.write(json.dumps(results))

    # landcover --------------------------
    # need to assign to each ij the classtype

    from img_utils import getRastervalue

    data["land_use"] = data.apply(getRastervalue,
                                  args=('../Data/Satellite/esa_landcover.tif',),
                                  axis=1)


if __name__ == '__main__':

    # Preload our model
    print("* Flask starting server...")

    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)