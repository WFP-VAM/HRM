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
from nn_extractor import NNExtractor
from utils import scoring_postprocess

from master_utils import download_score_merge

from sklearn.externals import joblib


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
    step = config["satellite_step"][0]
    provider = config["satellite_source"][0]
    start_date = config["sentinel_start"][0]
    end_date = config["sentinel_end"][0]
    land_use_raster = config["land_use_raster"][0]
    network_model = config['network_model'][0]
    custom_weights = config['custom_weights'][0]
    indicator_log = config['indicator_log'][0]
    model_pca = config['model_pca'][0]
    output = config['output'][0]
    model_grid_parameters = config['model_grid_parameters'][0]

    ## 1. Rasterize Country Shapefile

    country_shp="../Data/Geofiles/Shapefiles/ADM0/sen_admbnda_adm0_1m_gov_ocha_04082017/sen_admbnda_adm0_1m_gov_ocha_04082017.shp"
    cell_size=0.05
    no_data=-99
    output="../Data/Geofiles/Rasters/Senegal_raster_nodata_lowres.tif"

    #gdal_rasterize -a_nodata -99 -burn 1 -tr 0.05 0.05 -l sen_admbnda_adm0_1m_gov_ocha_04082017 "/Users/pasquierjb/Google Drive/WFP_Shared/Projects/HRM/Data/Shapefiles/ADM0/sen_admbnda_adm0_1m_gov_ocha_04082017/sen_admbnda_adm0_1m_gov_ocha_04082017.shp" /Users/pasquierjb/Desktop/test6.tif

    ## 2. Create list of i,j

    #raster="../Data/Geofiles/Rasters/Senegal_raster_nodata.tif"
    src = rasterio.open(raster)
    list_j, list_i = np.where(src.read()[0] != src.nodata)

    src.close()

    ## 3. Download images

    GRID = RasterGrid(raster)

    for sat in provider.split(","):
        data = download_score_merge(data, GRID, list_i, list_j, raster, step, sat, start_date, end_date, network_model, custom_weights)

    X = features.drop(['index', 'i', 'j'], axis=1)
    clf = joblib.load('../Models/ridge_model_config_id_{}.pkl'.format(id))
    y_hat = clf.predict(X)

    outfile="../Data/Outputs/{}.tif".format(id)

    ds = gdal.Open(raster)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape
    arr_out = np.zeros(arr.shape)-99
    arr_out[list_j,list_i] = y_hat
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)

    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input

    outdata.GetRasterBand(1).SetNoDataValue(-99)
    outdata.GetRasterBand(1).WriteArray(arr_out)

    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None


if __name__ == "__main__":

    import tensorflow as tf
    for id in sys.argv[1:]:
        run(id)

    # rubbish collection
    tf.keras.backend.clear_session()
