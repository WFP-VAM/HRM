from img_lib import RasterGrid
import pandas as pd
import numpy as np
from flask import send_file
import os


def downscale(config, request):

    country = request.form['country']
    algorithm = request.form['algorithm']
    file = request.files['file']

    # country ----------------------------------------------
    raster = '{}_0.01_4326_1.tif'.format(country)
    local_raster = 'temp/'+raster
    print('-> getting raster ', raster)
    # download from AWS S3
    import boto3
    bucket_name = config['rasters_bucket']
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).download_file(raster, local_raster)
    print('-> raster loaded.')

    # load dataset -----------------------------------------
    print('-> loading dataset from input form...')
    data = pd.read_csv(file)

    # load relative raster
    print('-> loading raster ', local_raster)
    GRID = RasterGrid(local_raster)
    try:
        data['i'], data['j'] = GRID.get_gridcoordinates(data)
    except IndexError:
        print('ERROR: raster and data are not from the same country!')
        raise
    # ------------------------------------

    # Grouping clusters that belong to the same tile.
    cluster_N = 'countbyEA'
    print("Number of clusters: {} ".format(len(data)))

    def wavg(g, df, weight_series):
        w = df.ix[g.index][weight_series]
        return (g * w).sum() / w.sum()

    import functools
    fnc = functools.partial(wavg, df=data, weight_series=cluster_N)

    try:
        data = data.groupby(["i", "j"]).agg({'Indicator': fnc, 'gpsLatitude': fnc, 'gpsLongitude': fnc}).reset_index()
    except:
        print("No weights, taking the average per i and j")
        data = data[['gpsLatitude', 'gpsLongitude', 'Indicator']].groupby(["i", "j"]).mean().reset_index()

    print("Number of unique tiles: {} ".format(len(data)))

    # train model ------------------------------------
    X = pd.DataFrame({"i": data["i"], "j": data["j"]})
    y = data.Indicator.values

    from model import IndicatorScaler

    model = IndicatorScaler(algorithm, X, y)

    # all country predictions ------------
    print('-> loading all grid points in the country')
    import rasterio
    src = rasterio.open(local_raster)
    list_j, list_i = np.where(src.read()[0] > 0)
    src.close()

    # also add the gps coordinates to the data for later use
    coords_i, coords_j = GRID.get_gpscoordinates(list_i, list_j)
    res = pd.DataFrame({"i": list_i,
                        "j": list_j,
                        "gpsLongitude": coords_i,
                        "gpsLatitude": coords_j})

    # ------------------------------------

    # landcover --------------------------
    esa_raster = 'esa_landcover_{}.tif'.format(country)
    local_esa_raster = 'temp/'+esa_raster
    s3.Bucket(bucket_name).download_file(esa_raster, local_esa_raster)

    print('-> getting landuse from ESA ({})'.format(local_esa_raster))
    from img_utils import getRastervalue
    res = getRastervalue(res, local_esa_raster)
    # ------------------------------------

    # predictions for all data left -------
    print('-> running predictions...')
    res['yhat'] = model.model.predict(res[['i', 'j']])
    # ------------------------------------

    # saves to disk ---------------------
    # no idea how this works
    from exporter import tifgenerator
    outfile = "temp/scalerout_{}_{}.tif".format(country, algorithm)
    tifgenerator(outfile=outfile,
                 raster_path=local_raster,
                 df=res)
    # -------------------------------------

    print('-> return file to client.')
    return send_file('../'+outfile,
                     mimetype='image/tiff',
                     as_attachment=True,
                     attachment_filename=country + "_" + algorithm + ".tif")
