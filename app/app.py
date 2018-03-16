from flask import Flask, render_template, request
import pandas as pd
import yaml
import numpy as np
import datetime
import json
import os
import sys
sys.path.append(os.path.join("..","Src"))
from img_lib import RasterGrid

app = Flask(__name__, instance_relative_config=True)

try:
    config_file = '../app_config.yml'
    with open(config_file, 'r') as cfgfile:
        config = yaml.load(cfgfile)
except FileNotFoundError:
    print('config file {} not found.'.format(config_file))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/downscale', methods=['POST'])
def master():

    # parameters
    raster_path = '../Data/Geofiles/Senegal_0.01_4326_1.tif'
    results_path = "../app/data/config_id_KNN.csv"

    # country ----------------------------------------------
    country_txt = request.form['country']

    # load dataset -----------------------------------------
    print('-> loading dataset from input form...')
    data = pd.read_csv(request.files['file'])

    # load relative raster

    print('-> loading raster from disk: ', raster_path)
    GRID = RasterGrid(raster_path)
    try:
        data['i'], data['j'] = GRID.get_gridcoordinates(data)
    except IndexError:
        print('ERROR: raster and data are not from the same country!')
        raise
    # ------------------------------------

    # group stuff, but is ugly

    # train KNN---------------------------------------
    X = pd.DataFrame({"i": data["i"], "j": data["j"]})
    y = data.Indicator.values

    # libs
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
    from evaluation_utils import MAPE, r2_pearson

    print('-> 5 folds cross validation and grid searching...')

    outer_cv = KFold(5, shuffle=True, random_state=75788)
    inner_cv = KFold(5, shuffle=True, random_state=1673)

    parameters = {'n_neighbors': range(1, 20)}

    model = KNeighborsRegressor(weights='distance')
    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2_pearson)
    # ------------------------------------

    # evaluate ---------------------------
    score = cross_val_score(clf, X, y, scoring=r2_pearson, cv=outer_cv)
    score_MAPE = cross_val_score(clf, X, y, scoring=MAPE, cv=outer_cv)

    print('-> scores: ', score)
    # score
    results = {
        'score': score.mean(),
        'MAPE': score_MAPE.mean(),
        'time': str(datetime.datetime.now())
    }
    with open('results.txt', 'w') as file:
        file.write(json.dumps(results))
    print('-> scores written to disk.')
    # ------------------------------------

    # all country predictions ------------
    # train model on all data
    clf.fit(X, y)
    print('INFO: best parameter: ', clf.fit(X, y).best_params_)

    print('-> loading all grid points in the country')
    import rasterio
    src = rasterio.open(raster_path)
    list_j, list_i = np.where(src.read()[0] > 0)
    src.close()

    # also add the coordinates to the data for later use
    coords_i, coords_j = GRID.get_gpscoordinates(list_i, list_j)
    res = pd.DataFrame({"i": list_i,
                        "j": list_j,
                        "gpsLongitude": coords_i,
                        "gpsLatitude": coords_j})
    # ------------------------------------

    # landcover --------------------------
    esa_raster = '../Data/Geofiles/esa_landcover_senegal_cmp.tif'
    print('-> getting landuse from ESA ({})'.format(esa_raster))
    from img_utils import getRastervalue
    res = getRastervalue(res, esa_raster)
    # ------------------------------------

    # predictions for all data left -------
    print('-> running predictions...')
    res['yhat'] = clf.predict(res[['i', 'j']])
    print('-> writing results to: ', results_path)
    res.to_csv(results_path, index=False)
    # ------------------------------------

    # saves to disk ---------------------
    # no idea how this works
    import gdal
    outfile = "../app/data/scalerout_{}_KNN.tif".format(request.form['country'])
    print('-> writing: ', outfile)
    # create empty raster from the original one
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape
    arr_out = np.zeros(arr.shape) - 99
    arr_out[res['j'], res['i']] = res['yhat']
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)

    outdata.SetGeoTransform(ds.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input

    outdata.GetRasterBand(1).SetNoDataValue(-99)
    outdata.GetRasterBand(1).WriteArray(arr_out)

    outdata.FlushCache()  # saves to disk!!
    # -------------------------------------
    return '-> DONE.'


if __name__ == '__main__':

    # Preload our model
    print("* Flask starting server...")

    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)