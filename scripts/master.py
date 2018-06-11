"""
- loads the survey data (already preprocessed)
- donwloads the relevant satellite images
- extracts the features with a pre-trained net
- trains a regression model to predict food insecurity
"""
import os
import sys
from sqlalchemy import create_engine
import yaml
import pandas as pd
import numpy as np
import functools
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..", "Src"))
from img_lib import RasterGrid
from nn_extractor import NNExtractor
from osm import OSM_extractor
from sentinel_utils import gee_sentinel_raster, gee_raster_mean
from utils import squaretogeojson, date_range, df_boundaries, points_to_polygon


def run(id):
    # ----------------- #
    # SETUP #############
    # ----------------- #

    print(str(np.datetime64('now')), " INFO: config id =", id)

    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    config = pd.read_sql_query("select * from config_new where id = {}".format(id), engine)
    dataset = config.get("dataset_filename")[0]
    indicator = config["indicator"][0]
    raster = config["satellite_grid"][0]
    aggregate_factor = config["aggregation"][0]

    # ----------------------------------- #
    # WorldPop Raster too fine, aggregate #
    from utils import aggregate
    if aggregate_factor > 1:
        print('INFO: aggregating raster ...')
        base_raster = "../tmp/local_raster.tif"
        aggregate(raster, base_raster, aggregate_factor)
    else:
        base_raster = raster

    nightlights_date_start = config["nightlights_date"][0].get("start")
    nightlights_date_end = config["nightlights_date"][0].get("end")

    if config['satellite_config'][0].get('satellite_images') == 'Y':
        step = config['satellite_config'][0].get("satellite_step")

    # -------- #
    # DATAPREP #
    # -------- #
    data = pd.read_csv(dataset)
    data_cols = data.columns.values

    # grid
    GRID = RasterGrid(base_raster)
    list_i, list_j = GRID.get_gridcoordinates(data)

    # to use the centroid from the tile instead
    # coords_x, coords_y = np.round(GRID.get_gpscoordinates(list_i, list_j), 5)
    #data['gpsLongitude'], data['gpsLatitude'] = coords_x, coords_y
    coords_x, coords_y = np.round(GRID.get_gpscoordinates(list_i, list_j), 5)

    # OPTIONAL: REPLACING THE CLUSTER COORDINATES BY THE CORRESPONDING GRID CENTER COORDINATES
    # data['gpsLongitude'], data['gpsLatitude'] = coords_x, coords_y

    data["i"], data["j"] = list_i, list_j

    # Get Polygon Geojson of the boundaries
    minlat, maxlat, minlon, maxlon = df_boundaries(data, buffer=0.05, lat_col="gpsLatitude", lon_col="gpsLongitude")
    area = points_to_polygon(minlon, minlat, maxlon, maxlat)

    # --------------------------- #
    # GROUP CLUSTERS IN SAME TILE #
    # --------------------------- #
    # TODO: looks like shit
    cluster_N = 'n'
    print("Number of clusters: {} ".format(len(data)))

    # def wavg(g, df, weight_series):
    #     w = df.ix[g.index][weight_series]
    #     return (g * w).sum() / w.sum()
    #
    # fnc = functools.partial(wavg, df=data, weight_series=cluster_N)
    #
    # try:
    #     data = data.groupby(["i", "j"]).agg({indicator: fnc, 'gpsLatitude': fnc, 'gpsLongitude': fnc}).reset_index()
    # except KeyError:
    #     print("No weights, taking the average per i and j")
    #     data = data[['i', 'j', 'n', 'gpsLatitude', 'gpsLongitude', indicator]].groupby(["i", "j"]).mean().reset_index()
    #
    # print("Number of unique tiles: {} ".format(len(data)))

    list_i, list_j, pipeline = data["i"], data["j"], 'evaluation'

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

            if os.path.exists("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline)):
                print('INFO: already scored.')
                features = pd.read_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline))
            else:
                print('INFO: scoring ...')
                # extract the features
                network = NNExtractor(id, sat, GRID.image_dir, sat, step, GRID)
                print('INFO: extractor instantiated.')

                features = network.extract_features(list_i, list_j, sat, start_date, end_date, pipeline)
                # normalize the features

                features.to_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline), index=False)

            features = features.drop('index', 1)
            data = data.merge(features, on=["i", "j"])

        data.to_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(id), index=False)

        print('INFO: features extracted.')

    # --------------- #
    # add nightlights #
    # --------------- #

    from nightlights import Nightlights

    NGT = Nightlights(area, '../Data/Geofiles/nightlights/', nightlights_date_start, nightlights_date_end)
    data['nightlights'] = NGT.nightlights_values(data)

    # ---------------- #
    # add OSM features #
    # ---------------- #
    OSM = OSM_extractor(data)
    tags = {"amenity": ["school", "hospital"], "natural": ["tree"]}
    osm_gdf = {}
    osm_features = []

    for key, values in tags.items():
        for value in values:
            osm_gdf["value"] = OSM.download(key, value)
            osm_tree = OSM.gpd_to_tree(osm_gdf["value"])
            dist = data.apply(OSM.distance_to_nearest, args=(osm_tree,), axis=1)
            #density = data.apply(OSM.density, args=(osm_gdf["value"],), axis=1)
            data['distance_{}'.format(value)] = dist.apply(lambda x: np.log(0.0001 + x))
            osm_features.append('distance_{}'.format(value))
            #data['density_{}'.format(value)] = density.apply(lambda x: np.log(0.0001 + x))
            #osm_features.append('density_{}'.format(value))

    # ---------------- #
    #   NDBI,NDVI,NDWI #
    # ---------------- #
    # TODO: Use efficiently maxNDBImaxNDVImaxNDWI_sum_todf
    print('INFO: getting NDBI, NDVI, NDWI ...')

    start_date = "2017-01-01"  # TODO: Add to config, be careful no image before 2015
    end_date = "2018-01-01"
    for i in date_range(start_date, end_date, 3):
        print('INFO: getting max NDVI between dates: {}'.format(i))
        gee_ndvi_max_raster = gee_sentinel_raster(i[0], i[1], area, ind="NDVI")
        data["max_NDVI_{}_{}".format(i[0], i[1])] = data.apply(gee_raster_mean, args=(gee_ndvi_max_raster, "gpsLatitude", "gpsLongitude", "NDVI"), axis=1)

    print('INFO: getting max NDBI')
    gee_ndbi_max_raster = gee_sentinel_raster(start_date, end_date, area, ind="NDBI")
    data["max_NDBI"] = data.apply(gee_raster_mean, args=(gee_ndbi_max_raster, "gpsLatitude", "gpsLongitude", "NDBI"), axis=1)

    print('INFO: getting max NDWI')
    gee_ndwi_max_raster = gee_sentinel_raster(start_date, end_date, area, ind="NDWI")
    data["max_NDWI"] = data.apply(gee_raster_mean, args=(gee_ndwi_max_raster, "gpsLatitude", "gpsLongitude", "NDWI"), axis=1)

    # --------------- #
    # save features   #
    # --------------- #

    features_list = list(set(data.columns) - set(data_cols) - set(['i', 'j']))

    # Standardize Features (0 mean and 1 std)
    data[features_list] = (data[features_list] - data[features_list].mean()) / data[features_list].std()

    data.to_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(id), index=False)

    # --------------- #
    # model indicator #
    # --------------- #
    data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  # shuffle data

    data_features = data[features_list]

    # if take log of indicator
    if config['log'][0]:
        data[indicator] = np.log(data[indicator])
    from modeller import Modeller
    md = Modeller(['kNN', 'Kriging', 'RmSense', 'Ensamble'], data_features)
    cv_loops = 20
    md.compute(data[['i', 'j']], data[indicator].values, cv_loops)

    # save model for production
    md.save_models(id)
    print(str(np.datetime64('now')), 'INFO: model saved.')

    # ------------------ #
    # write scores to DB #
    # ------------------ #

    r2, r2_var = np.mean(md.scores['Ensamble']), np.var(md.scores['Ensamble'])
    r2_knn, r2_var_knn = np.mean(md.scores['kNN']), np.var(md.scores['kNN'])
    r2_rmsense, r2_var_rmsense = np.mean(md.scores['RmSense']), np.var(md.scores['RmSense'])
    y_duplicated = np.repeat(data[indicator], cv_loops)
    mape_rmsense = np.mean(np.abs([item for sublist in md.results['RmSense'] for item in sublist] - y_duplicated) / y_duplicated)
    if mape_rmsense == float("inf") or mape_rmsense == float("-inf"):
        mape_rmsense = 0

    query = """
    insert into results_new (run_date, config_id, r2, r2_var, r2_knn, r2_var_knn, r2_features, r2_var_features, mape_rmsense)
    values (current_date, {}, {}, {}, {}, {}, {}, {}, {}) """.format(
        config['id'][0],
        r2, r2_var, r2_knn, r2_var_knn, r2_rmsense, r2_var_rmsense, mape_rmsense)
    engine.execute(query)

    # ------------------------- #
    # write predictions to file #
    # ------------------------- #
    print('INFO: writing predictions to disk ...')
    results = pd.DataFrame({
        #'yhat': [item for sublist in md.results['kNN'] for item in sublist],
        'y': data[indicator].values,
        'lat': data['gpsLatitude'],
        'lon': data['gpsLongitude']})
    results.to_csv('../Data/Results/config_{}.csv'.format(id), index=False)


if __name__ == "__main__":

    import tensorflow as tf
    for id in sys.argv[1:]:
        run(id)

    # rubbish collection
    tf.keras.backend.clear_session()
