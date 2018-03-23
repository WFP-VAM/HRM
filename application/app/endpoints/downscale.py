from img_lib import RasterGrid
import pandas as pd
import numpy as np
from flask import send_file
import os


def downscale(config, request):

    country = request.form['country']
    algorithm = request.form['algorithm']

    # country ----------------------------------------------
    raster = '../{}_0.01_4326_1.tif'.format(country)
    print('-> getting raster ', raster)
    # download from AWS S3
    import boto3
    bucket_name = config['rasters_bucket']
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).download_file(raster, raster)
    print('raster loaded.')

    # load dataset -----------------------------------------
    print('-> loading dataset from input form...')
    data = pd.read_csv(request.files['file'])

    # load relative raster
    print('-> loading raster ', raster)
    GRID = RasterGrid(raster)
    try:
        data['i'], data['j'] = GRID.get_gridcoordinates(data)
    except IndexError:
        print('ERROR: raster and data are not from the same country!')
        raise
    # ------------------------------------

    # train model ------------------------------------
    X = pd.DataFrame({"i": data["i"], "j": data["j"]})
    y = data.Indicator.values

    from model import IndicatorScaler

    model = IndicatorScaler(algorithm, X, y)

    # all country predictions ------------
    print('-> loading all grid points in the country')
    import rasterio
    src = rasterio.open(raster)
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
    esa_raster = '../data/esa_landcover_{}.tif'.format(country)
    s3.Bucket(bucket_name).download_file(esa_raster, esa_raster)

    print('-> getting landuse from ESA ({})'.format(esa_raster))
    from img_utils import getRastervalue
    res = getRastervalue(res, esa_raster)
    # ------------------------------------

    # predictions for all data left -------
    print('-> running predictions...')
    res['yhat'] = model.model.predict(res[['i', 'j']])
    # ------------------------------------

    # saves to disk ---------------------
    # no idea how this works
    from exporter import tifgenerator
    outfile = os.path.join("../data", "scalerout_{}_{}.tif".format(country, algorithm))
    tifgenerator(outfile=outfile,
                 raster_path=raster,
                 df=res)
    # -------------------------------------

    print('-> return file to client.')
    return send_file(outfile,
                     mimetype='image/tiff',
                     as_attachment=True,
                     attachment_filename=country + "_" + algorithm + ".tif")