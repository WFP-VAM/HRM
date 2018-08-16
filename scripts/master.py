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
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..", "Src"))
from img_lib import RasterGrid
from nn_extractor import NNExtractor
from osm import OSM_extractor
from utils import df_boundaries, points_to_polygon


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
    aggregate_factor = config["base_raster_aggregation"][0]
    scope = config["scope"][0]
    nightlights_date_start, nightlights_date_end = config["nightlights_date"][0].get("start"), config["nightlights_date"][0].get("end")
    s2_date_start, s2_date_end = config["NDs_date"][0].get("start"), config["NDs_date"][0].get("end")
    if config['satellite_config'][0].get('satellite_images') == 'Y': step = config['satellite_config'][0].get("satellite_step")

    # ----------------------------------- #
    # WorldPop Raster too fine, aggregate #
    from utils import aggregate
    if aggregate_factor > 1:
        print('INFO: aggregating raster {}'.format(raster))
        base_raster = "../tmp/local_raster.tif"
        aggregate(raster, base_raster, aggregate_factor)
    else:
        base_raster = raster

    # -------- #
    # DATAPREP #
    # -------- #
    data = pd.read_csv(dataset)
    data_cols = data.columns.values

    # grid
    GRID = RasterGrid(base_raster)
    list_i, list_j = GRID.get_gridcoordinates(data)

    # OPTIONAL: REPLACING THE CLUSTER COORDINATES BY THE CORRESPONDING GRID CENTER COORDINATES
    # data['gpsLongitude'], data['gpsLatitude'] = coords_x, coords_y

    data["i"], data["j"] = list_i, list_j

    # Get Polygon Geojson of the boundaries
    minlat, maxlat, minlon, maxlon = df_boundaries(data, buffer=0.05, lat_col="gpsLatitude", lon_col="gpsLongitude")
    area = points_to_polygon(minlon, minlat, maxlon, maxlat)

    print("Number of clusters: {} ".format(len(data)))

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
    OSM = OSM_extractor(minlon, minlat, maxlon, maxlat)
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

    S2 = S2indexes(area, '../Data/Geofiles/NDs/', s2_date_start, s2_date_end, scope)
    S2.download()
    data[['max_NDVI', 'max_NDBI', 'max_NDWI']] = S2.rms_values(data).apply(pd.Series)
    # --------------- #
    # save features   #
    # --------------- #
    # features to be use in the linear model
    features_list = list(sorted(set(data.columns) - set(data_cols) - set(['i', 'j'])))

    # Standardize Features (0 mean and 1 std)
    data[features_list] = (data[features_list] - data[features_list].mean()) / data[features_list].std()

    data.to_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(id), index=False)

    # --------------- #
    # model indicator #
    # --------------- #
    # shuffle dataset
    data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  # shuffle data

    # if set in the config, take log of indicator
    if config['log'][0]:
        data[indicator] = np.log(data[indicator])

    from modeller import Modeller
    X, y = data[features_list + ["gpsLatitude", "gpsLongitude"]], data[indicator]
    modeller = Modeller(X, rs_features=features_list, spatial_features=["gpsLatitude", "gpsLongitude"], scoring='r2', cv_loops=20)

    kNN_pipeline = modeller.make_model_pipeline('kNN')
    kNN_scores = modeller.compute_scores(kNN_pipeline, y)
    kNN_R2_mean = kNN_scores.mean()
    kNN_R2_std = kNN_scores.std()
    print("kNN_R2_mean: ", kNN_R2_mean, "kNN_R2_std: ", kNN_R2_std)

    Ridge_pipeline = modeller.make_model_pipeline('Ridge')
    Ridge_scores = modeller.compute_scores(Ridge_pipeline, y)
    Ridge_R2_mean = Ridge_scores.mean()
    Ridge_R2_std = Ridge_scores.std()
    print("Ridge_R2_mean: ", Ridge_R2_mean, "Ridge_R2_std: ", Ridge_R2_std)

    Ensemble_pipeline = modeller.make_ensemble_pipeline([kNN_pipeline, Ridge_pipeline])
    Ensemble_scores = modeller.compute_scores(Ensemble_pipeline, y)
    Ensemble_R2_mean = Ensemble_scores.mean()
    Ensemble_R2_std = Ensemble_scores.std()
    print("Ensemble_R2_mean: ", Ensemble_R2_mean, "Ensemble_R2_std: ", Ensemble_R2_std)

    # ------------------ #
    # write scores to DB #
    # ------------------ #

    query = """
    insert into results_new (run_date, config_id, r2, r2_sd, r2_knn, r2_sd_knn, r2_features, r2_sd_features, mape_rmsense)
    values (current_date, {}, {}, {}, {}, {}, {}, {}, {}) """.format(
        config['id'][0],
        Ensemble_R2_mean, Ensemble_R2_std, kNN_R2_mean, kNN_R2_std, Ridge_R2_mean, Ridge_R2_std, 0)
    engine.execute(query)

    # ------------------------- #
    # write predictions to file #
    # ------------------------- #

    print('INFO: writing predictions to disk ...')

    from sklearn.model_selection import cross_val_predict
    results = pd.DataFrame({
        'yhat': cross_val_predict(Ensemble_pipeline, X.values, y),
        'y': data[indicator].values,
        'lat': data['gpsLatitude'],
        'lon': data['gpsLongitude']})
    results.to_csv('../Data/Results/config_{}.csv'.format(id), index=False)

    # save model for production
    Ensemble_pipeline.fit(X.values, y)
    from sklearn.externals import joblib
    joblib.dump(Ensemble_pipeline, '../Models/Ensemble_model_config_id_{}.pkl'.format(id))
    print(str(np.datetime64('now')), 'INFO: model saved.')


if __name__ == "__main__":

    import tensorflow as tf
    for id in sys.argv[1:]:
        run(id)

    # rubbish collection
    tf.keras.backend.clear_session()
