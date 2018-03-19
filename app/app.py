from flask import Flask, render_template, request, send_file
import pandas as pd
import yaml
import numpy as np
import os
import sys
sys.path.append(os.path.join("..","Src"))
sys.path.append(os.path.join("..","app/src"))
from img_lib import RasterGrid

app = Flask(__name__, instance_relative_config=True)

try:
    config_file = '../app/app_config.yml'
    with open(config_file, 'r') as cfgfile:
        config = yaml.load(cfgfile)
except FileNotFoundError:
    print('config file {} not found.'.format(config_file))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/downscale', methods=['POST'])
def master():

    # country ----------------------------------------------
    raster_path = '../Data/Geofiles/'
    if request.form['country'] == 'Senegal':
        raster_path = raster_path + 'Senegal_0.01_4326_1.tif'
    elif request.form['country'] == 'Nigeria':
        raster_path = raster_path + 'Nigeria_0.01_4326_1.tif'
    else:
        raise print('Country not implemented')

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

    from model import IndicatorScaler

    model = IndicatorScaler('kNN', X, y)


    # all country predictions ------------
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
    esa_raster = '../Data/Geofiles/'
    if request.form['country'] == 'Nigeria':
        esa_raster = esa_raster + 'esa_landcover_nigeria_cmp.tif'
    elif request.form['country'] == 'Senegal':
        esa_raster = esa_raster + 'esa_landcover_senegal_cmp.tif'


    print('-> getting landuse from ESA ({})'.format(esa_raster))
    from img_utils import getRastervalue
    res = getRastervalue(res, esa_raster)
    # ------------------------------------

    # predictions for all data left -------
    print('-> running predictions...')
    res['yhat'] = model.model.predict(res[['i', 'j']])
    print('-> writing results to: ', config_file['results_path'])
    res.to_csv(config_file['results_path'], index=False)
    # ------------------------------------

    # saves to disk ---------------------
    # no idea how this works
    from exporter import tifgenerator
    outfile = "../app/data/scalerout_{}_KNN.tif".format(request.form['country'])
    tifgenerator(outfile=outfile,
                 raster_path=raster_path,
                 df=res)
    # -------------------------------------
    return send_file(outfile,
                     mimetype='image/tiff',
                     as_attachment=True,
                     attachment_filename=request.form['country']+".tif")


if __name__ == '__main__':

    # Preload our model
    print("* Flask starting server...")

    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)