# -*- coding: utf-8 -*-
"""
- downloads the pictures relevant for scoring
- extracts features
- loads a pre-trained model
- makes predictions
- plots
"""
import os
import sys
sys.path.append(os.path.join("..", "Src"))
from img_lib import RasterGrid
from sqlalchemy import create_engine
import yaml
import pandas as pd
import numpy as np
from nn_extractor import NNExtractor
from utils import tifgenerator, aggregate
from sklearn.externals import joblib
import click
import rasterio
from rasterio.mask import mask

# ---------- #
# PARAMETERS #
@click.command()
@click.option('--top_left', default=(15.173283, -4.293467))
@click.option('--bottom_left', default=(15.168365, -4.479364))
@click.option('--bottom_right', default=(15.448367, -4.506647))
@click.option('--top_right', default=(15.372248, -4.283885))
@click.option('--config_id', default=1)
def main(top_left, bottom_left, bottom_right, top_right, config_id):

    # ------#
    # SETUP #
    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    # connect to db and read config table
    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    config = pd.read_sql_query("select * from config_new where id = {}".format(config_id), engine)

    step = config["satellite_step"][0]
    start_date = config["sentinel_config"][0]["start_date"]
    end_date = config["sentinel_config"][0]["end_date"]
    base_raster = "../tmp/local_raster.tif"

    # ----------------------------------- #
    # WorldPop Raster too fine, aggregate #
    aggregate(config["satellite_grid"][0], base_raster, 10)

    # -------------------  #
    # CLIP RASTER TO SCOPE #
    geoms = [{'type': 'Polygon', 'coordinates': [[top_left, bottom_left, bottom_right, top_right]]}]

    with rasterio.open(base_raster) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

    # save the resulting raster
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform
                     })

    with rasterio.open(base_raster, "w", **out_meta) as dest:
        dest.write(out_image)

    # load the new clipped raster to the img_lib
    GRID = RasterGrid(base_raster)
    with rasterio.open(base_raster) as src:
        list_j, list_i = np.where(src.read()[0] != 0)
    print("INFO: downloading images in scope ...")
    coords_x, coords_y = np.round(GRID.get_gpscoordinates(list_i, list_j),5)

    # ------------------------------------------------------------- #
    # download images from Google and Sentinel and Extract Features #
    # ------------------------------------------------------------- #
    for sat in ['Google', 'Sentinel']:
        print('INFO: routine for provider: ', sat)
        # dopwnlaod the images from the relevant API
        GRID.download_images(list_i, list_j, step, sat, start_date, end_date)
        print('INFO: images downloaded.')

        print('INFO: scoring ...')
        # extarct the features
        network = NNExtractor(id, sat, GRID.image_dir, sat, step, GRID)
        print('INFO: extractor instantiated.')
        features = network.extract_features(list_i, list_j, sat, start_date, end_date, pipeline='scoring')
        # normalize the features
        features.to_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, config_id, 'scoring'), index=False)

    g_features = pd.read_csv("../Data/Features/features_{}_id_{}_{}.csv".format("Google", config_id, 'scoring'))
    s_features = pd.read_csv("../Data/Features/features_{}_id_{}_{}.csv".format("Sentinel", config_id, 'scoring'))

    data = pd.merge(g_features, s_features, on=['i','j', 'index'])
    data.to_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(config_id), index=False)

    print('INFO: features extracted.')
    # ---------------------- #
    # LOAD MODEL AND PREDICT #
    print("INFO: load model and predict ...")
    X = data.drop(['index', 'i', 'j'], axis=1)
    # load model and predict
    try:
        RmSense = joblib.load('../Models/RmSense_model_config_id_{}.pkl'.format(config_id))
        kNN = joblib.load('../Models/kNN_model_config_id_{}.pkl'.format(config_id))
    except FileNotFoundError:
        print('ERROR: model not found')

    yhat = (RmSense.predict(X) + kNN.predict(data[['i','j']])) / 2.
    results = pd.DataFrame({'i': list_i, 'j': list_j, 'lat': coords_y, 'lon': coords_x, 'yhat': yhat})

    outfile = "../Data/Results/scalerout_{}.tif".format(config_id)
    tifgenerator(outfile=outfile,
                 raster_path=base_raster,
                 df=results)


if __name__ == '__main__':

    main()
