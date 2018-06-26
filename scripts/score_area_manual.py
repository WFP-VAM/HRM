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
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..","Src"))
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
from osm import OSM_extractor

# ---------- #
# PARAMETERS #
@click.command()
@click.option('--top_left', default=(15.224515, -4.292011))
@click.option('--bottom_left', default=(15.214246, -4.492340))
@click.option('--bottom_right', default=(15.444959, -4.449897))
@click.option('--top_right', default=(15.462812, -4.292011))
@click.option('--config_id')
def main(top_left, bottom_left, bottom_right, top_right, config_id):

    print(str(np.datetime64('now')), " INFO: config id =", config_id)

    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    config = pd.read_sql_query("select * from config_new where id = {}".format(config_id), engine)
    dataset = config.get("dataset_filename")[0]
    raster = config["satellite_grid"][0]
    aggregate_factor = 10 #config["base_raster_aggregation"][0]
    scope = config["scope"][0]
    nightlights_date_start, nightlights_date_end = config["nightlights_date"][0].get("start"), \
                                                   config["nightlights_date"][0].get("end")
    s2_date_start, s2_date_end = config["NDs_date"][0].get("start"), config["NDs_date"][0].get("end")
    if config['satellite_config'][0].get('satellite_images') == 'Y': step = config['satellite_config'][0].get(
        "satellite_step")

    # ----------------------------------- #
    # WorldPop Raster too fine, aggregate #

    if aggregate_factor > 1:
        print('INFO: aggregating raster ...')
        base_raster = "../tmp/local_raster.tif"
        aggregate(raster, base_raster, aggregate_factor)
    else:
        base_raster = raster

    # ---------------- #
    # AREA OF INTEREST #
    # ---------------- #
    dataset_df = pd.read_csv(dataset)
    data_cols = dataset_df.columns.values

    # create geometry
    aoi = [{'type': 'Polygon', 'coordinates': [[top_left, bottom_left, bottom_right, top_right]]}]

    # crop raster
    with rasterio.open(base_raster) as src:
        out_image, out_transform = mask(src, aoi, crop=True)
        out_meta = src.meta.copy()

    # save the resulting raster
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform
                     })

    final_raster = "../tmp/final_raster.tif"
    with rasterio.open(final_raster, "w", **out_meta) as dest:
        dest.write(out_image)
        list_j, list_i = np.where(dest.read()[0] != dest.nodata)

    # instantiate GRID
    GRID = RasterGrid(final_raster)

    coords_x, coords_y = np.round(GRID.get_gpscoordinates(list_i, list_j), 5)

    data = pd.DataFrame({"i": list_i, "j": list_j})
    data["gpsLatitude"] = coords_y
    data["gpsLongitude"] = coords_x

    print("Number of clusters: {} ".format(len(data)))

    list_i, list_j, pipeline = data["i"], data["j"], 'scoring'

    # ------------------------------------------------------------- #
    # download images from Google and Sentinel and Extract Features #
    # ------------------------------------------------------------- #
    if config["satellite_config"][0]["satellite_images"] != 'N':

        start_date = config["satellite_config"][0]["start_date"]
        end_date = config["satellite_config"][0]["end_date"]

        for sat in ['Google', 'Sentinel']:
            print('INFO: routine for provider: ', sat)
            # downlaod the images from the relevant API
            GRID.download_images(list_i, list_j, step, sat, start_date, end_date, zoom_vhr=16, img_size_sentinel=5000)
            print('INFO: images downloaded.')

            if os.path.exists("../Data/Features/features_{}_id_{}_{}.csv".format(sat, config_id, pipeline)):
                print('INFO: already scored.')
                features = pd.read_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, config_id, pipeline))
            else:
                print('INFO: scoring ...')
                # extract the features
                network = NNExtractor(id, sat, GRID.image_dir, sat, step, GRID)
                print('INFO: extractor instantiated.')

                features = network.extract_features(list_i, list_j, sat, start_date, end_date, pipeline)
                # normalize the features

                features.to_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, config_id, pipeline), index=False)

            features = features.drop('index', 1)
            data = data.merge(features, on=["i", "j"])

        data.to_csv("../Data/Features/features_all_id_{}_{}.csv".format(config_id, pipeline), index=False)

        print('INFO: features extracted.')

    # --------------- #
    # add nightlights #
    # --------------- #
    from nightlights import Nightlights
    from geojson import Polygon

    NGT = Nightlights(Polygon([[top_left, bottom_left, bottom_right, top_right]]), '../Data/Geofiles/nightlights/', nightlights_date_start, nightlights_date_end)
    data['nightlights'] = NGT.nightlights_values(data)

    # ---------------- #
    # add OSM features #
    # ---------------- #
    OSM = OSM_extractor(dataset_df)
    tags = {"amenity": ["school", "hospital"], "natural": ["tree"]}
    osm_gdf = {}
    osm_features = []

    for key, values in tags.items():
        for value in values:
            osm_gdf["value"] = OSM.download(key, value)
            osm_tree = OSM.gpd_to_tree(osm_gdf["value"])
            dist = data.apply(OSM.distance_to_nearest, args=(osm_tree,), axis=1)
            data['distance_{}'.format(value)] = dist.apply(lambda x: np.log(0.0001 + x))
            osm_features.append('distance_{}'.format(value))

    # ---------------- #
    #   NDBI,NDVI,NDWI #
    # ---------------- #
    print('INFO: getting NDBI, NDVI, NDWI ...')

    from rms_indexes import S2indexes

    S2 = S2indexes(Polygon([[top_left, bottom_left, bottom_right, top_right]]), '../Data/Geofiles/NDs/', s2_date_start, s2_date_end, scope)
    S2.download()
    data[['max_NDVI', 'max_NDBI', 'max_NDWI']] = S2.rms_values(data).apply(pd.Series)

    # --------------- #
    # save features   #
    # --------------- #

    features_list = list(sorted(set(data.columns) - set(data_cols) - set(['i', 'j'])))

    # Standardize Features (0 mean and 1 std)
    data[features_list] = (data[features_list] - data[features_list].mean()) / data[features_list].std()

    data.to_csv("../Data/Features/features_all_id_{}_{}.csv".format(config_id, pipeline), index=False)

    # Open model
    ensemble_pipeline = joblib.load('../Models/Ensemble_model_config_id_{}.pkl'.format(config_id))
    print(str(np.datetime64('now')), 'INFO: model loaded.')

    X = data[features_list + ["gpsLatitude", "gpsLongitude"]]
    ensemble_predictions = ensemble_pipeline.predict(X.values)

    # if take log of indicator
    if config['log'][0]:
        ensemble_predictions = np.exp(ensemble_predictions)
    print('list: ', len(list_i), len(list_j), ' coords: ', len(coords_x), len(coords_x), ' preds: ', len(ensemble_predictions))
    #results = pd.DataFrame({'i': list_i, 'j': list_j, 'lat': coords_y, 'lon': coords_x, 'yhat': ensemble_predictions})
    data['yhat'] = ensemble_predictions
    outfile = "../Data/Results/scalerout_{}.tif".format(config_id)
    tifgenerator(outfile=outfile,
                 raster_path=final_raster,
                 df=data)


if __name__ == '__main__':

    main()
