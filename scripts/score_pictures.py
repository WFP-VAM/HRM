# -*- coding: utf-8 -*-
"""
- downloads the pictures relevant for scoring
- extracts features
- loads a pre-trained model
- makes predictions
"""
import os
import sys
sys.path.append(os.path.join("..", "Src"))
from img_lib import RasterGrid
from sqlalchemy import create_engine
import yaml
import pandas as pd
from nn_extractor import NNExtractor
from utils import scoring_postprocess, get_coordinates_from_shp, zonal_stats, shape2json
from sklearn.externals import joblib
import json
import numpy as np
from branca.colormap import linear
import click
import folium


# ---------- #
# PARAMETERS #
@click.command()
@click.option('--adm0_file_path', type=click.Path(exists=True), default="../Data/Shapefiles/UGA_adm_shp/UGA_adm0.shp")
@click.option('--path_to_shapefile', type=click.Path(exists=True), default="../Data/Shapefiles/UGA_adm_shp/UGA_adm1.shp")
@click.option('--config_id', default=1)
@click.option('--gpscoordinates_sampling', default=0.1)
@click.option('--adm', default=1)
def main(adm0_file_path, path_to_shapefile, config_id, gpscoordinates_sampling, adm):

    # ------#
    # SETUP #
    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    # connect to db and read config table
    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    config = pd.read_sql_query("select * from config where id = {}".format(config_id), engine)

    # get raster
    image_dir = os.path.join("../Data", "Satellite", config["satellite_source"][0])
    GRID = RasterGrid(config["satellite_grid"][0], image_dir)

    # ----------------#
    # IMAGES IN SCOPE #
    print("INFO: downloading images in scope ...")
    adm0_lat, adm0_lon = get_coordinates_from_shp(adm0_file_path, spacing=gpscoordinates_sampling)
    # get the matching tiles
    list_i, list_j = GRID.get_gridcoordinates2(adm0_lat, adm0_lon)
    # download images to predict
    GRID.download_images(list_i, list_j, step=config['satellite_step'][0], provider=config['satellite_source'][0])

    # ---------------------------------- #
    # EXTRACT FEATURES FOR IMGS IN SCOPE #
    print("INFO: extracting features for images in scope ...")
    network = NNExtractor(image_dir, 'ResNet50', step=config['satellite_step'][0])
    features = scoring_postprocess(network.extract_features(list_i, list_j))

    # ---------------------- #
    # LOAD MODEL AND PREDICT #
    print("INFO: load model and predict ...")
    X = features.drop(['index', 'i', 'j'], axis=1)
    # load model and predict
    try:
        clf = joblib.load('../Models/ridge_model_config_id_{}.pkl'.format(config_id))
    except FileNotFoundError:
        print('ERROR: model not found')

    yhat = clf.predict(X)
    results = pd.DataFrame({'lat': adm0_lat, 'lon': adm0_lon, 'yhat': yhat})

    # ---------------------- #
    # COMPUTE MEANS FOR ADM2 #
    print("INFO: computing means for lower adm level ...")
    polygon_means = zonal_stats(path_to_shapefile, results['lon'], results['lat'], results['yhat'])

    # ---- #
    # PLOT #
    # convert shapefile to GeoJSON
    print("INFO: plotting ...")
    shape2json(path_to_shapefile, outfile=path_to_shapefile[:-3]+"json")

    with open(path_to_shapefile[:-3]+"json") as f:
        adm2_map = json.load(f)

    map_df = pd.DataFrame({'loc': [adm2_map['features'][i]['id'] for i in range(0,len(adm2_map['features']))],
                           'means': polygon_means})

    map_df.means = map_df.means.fillna(np.mean(map_df.means))  # TODO: handle nans
    map_dict = map_df.set_index('loc')['means']

    colormap = linear.YlGn.scale(
        map_df.means.min(),
        map_df.means.max())

    m = folium.Map(
        location=[1.130956, 32.354771],
        tiles='Stamen Terrain',
        zoom_start=6
    )
    folium.GeoJson(
        adm2_map,
        name='map_df',
        style_function=lambda feature: {
            'fillColor': colormap(map_dict[feature['id']]),
            'color': 'black',
            'weight': 0.5,
            'dashArray': '5, 5',
            'fillOpacity': 0.7,
        }
    ).add_to(m)
    folium.LayerControl().add_to(m)
    colormap.caption = 'indicator color scale'
    colormap.add_to(m)

    # --------- #
    # SAVE PLOT #
    print("INFO: saving plot ...")
    if not os.path.exists('../Plots'):
        os.makedirs('../Plots')
    m.save(os.path.join('../Plots','map_config_{}_amd_{}.html'.format(config_id, adm)))


if __name__ == '__main__':

    main()
