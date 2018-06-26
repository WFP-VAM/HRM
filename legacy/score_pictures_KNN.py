from sqlalchemy import create_engine
import yaml
import pandas as pd
import numpy as np

import rasterio
import gdal

import sys
import os
sys.path.append(os.path.join("..","Src"))
from img_lib import RasterGrid
from master_utils import download_score_merge


from evaluation_utils import MAPE, r2_pearson, r2
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict

def run(id):

    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                            private_config['DB']['host'], private_config['DB']['database']))

    config = pd.read_sql_query("select * from config where id = {}".format(id), engine)

    dataset = config["dataset_filename"][0]
    indicator = config["indicator"][0]
    raster = config["satellite_grid"][0]
    indicator_log = config['indicator_log'][0]


    ## load data

    GRID = RasterGrid(raster)
    list_i, list_j = GRID.get_gridcoordinates(dataset)#

    hh_data = pd.read_csv(dataset)

    data = hh_data
    data["i"] = list_i
    data["j"] = list_j

    cluster_N = 'countbyEA'

    try:
        data=data.groupby(["i","j"]).apply(lambda x: np.average(x[indicator],weights=x[cluster_N])).to_frame(name = indicator).reset_index()
    except:
        data=data.groupby(["i","j"]).mean()


    X = pd.DataFrame({"i":data["i"],"j":data["j"]})
    y = data[indicator].values

    # Log-normal distribution
    if indicator_log == True:
        y = np.log(y)

    # TRAIN MODEL
    outer_cv = KFold(5, shuffle=True, random_state=75788)
    inner_cv = KFold(5, shuffle=True, random_state=1673)
    print(str(np.datetime64('now')), " INFO: training model ...")

    from sklearn.neighbors import KNeighborsRegressor

    k = np.arange(20)+1
    parameters = {'n_neighbors': k}

    model = KNeighborsRegressor(weights='distance')
    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=inner_cv, scoring=r2_pearson)

    score = cross_val_score(clf, X, y, scoring=r2_pearson, cv=outer_cv)
    score_r2 = cross_val_score(clf, X, y, scoring=r2, cv=outer_cv)
    score_MAPE = cross_val_score(clf, X, y, scoring=MAPE, cv=outer_cv)

    print('INFO: Pearson score: ', score.mean())

    clf.fit(X,y)
    print('INFO: best parameter: ', clf.fit(X, y).best_params_)

    ##  Create list of i,j

    src = rasterio.open(raster)
    list_j, list_i = np.where(src.read()[0] != src.nodata)

    src.close()

    ## Score images

    X = pd.DataFrame({"i": list_i, "j": list_j})

    y_hat = clf.predict(X)

    outfile = "../Data/Outputs/config_id_{}_KNN.tif".format(id)

    ds = gdal.Open(raster)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape
    arr_out = np.zeros(arr.shape) - 99
    arr_out[list_j, list_i] = y_hat
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)

    outdata.SetGeoTransform(ds.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input

    outdata.GetRasterBand(1).SetNoDataValue(-99)
    outdata.GetRasterBand(1).WriteArray(arr_out)

    outdata.FlushCache() # saves to disk!!
    outdata = None
    band = None
    ds = None


if __name__ == "__main__":

    for id in sys.argv[1:]:
        run(id)
