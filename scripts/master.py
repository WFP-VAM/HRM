"""
- loads the survey data (already preprocessed)
- donwloads the relevant satellite images
- extracts the features with a pre-trained net
- trains a regression model to predict food insecurity
"""
import os
import sys
import pandas as pd
import numpy as np
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..", "Src"))
from base_layer import BaseLayer
from osm import OSM_extractor
from utils import boundaries, points_to_polygon, get_config_db, get_config_file, write_scores_to_file, write_scores_to_db


def run(file, id=None):
    # ----------------- #
    # SETUP #############
    # ----------------- #
    if id is None:
        config = get_config_file(file)
    else:
        config, engine = get_config_db(id)

    # --------------------- #
    # Setting up playground #
    # --------------------- #
    data = pd.read_csv(config['dataset_filename'])
    print(str(np.datetime64('now')), 'INFO: original dataset lenght: ', data.shape[0])
    data['gpsLongitude'] = np.round(data['gpsLongitude'], 5)
    data['gpsLatitude'] = np.round(data['gpsLatitude'], 5)

    # avoid duplicates
    data = data[['gpsLongitude', 'gpsLatitude', config['indicator']]].groupby(['gpsLongitude', 'gpsLatitude']).mean()

    # base layer
    GRID = BaseLayer(config['base_raster'], data.index.get_level_values('gpsLongitude'), data.index.get_level_values('gpsLatitude'))
    # TODO: we should enforce the most accurate i and j when training, i.e. aggregate = 1?

    # Get Polygon Geojson of the boundaries
    # TODO: maybe go into BaseLayer class?
    minlat, maxlat, minlon, maxlon = boundaries(GRID.lat, GRID.lon, buffer=0.05)
    area = points_to_polygon(minlon, minlat, maxlon, maxlat)

    print(str(np.datetime64('now')), "INFO: Number of clusters: {} ".format(len(data)))

    pipeline = 'evaluation'

    # ------------------------------- #
    # get features from Google images #
    # ------------------------------- #
    if config['satellite_config']['satellite_images'] in ['Y', 'G']:
        features_path = "../Data/Features/features_Google_id_{}_{}.csv".format(id, pipeline)
        data_path = "../Data/Satellite/"
        from google_images import GoogleImages

        if os.path.exists(features_path):
            print('INFO: already scored.')
            features = pd.read_csv(
                features_path.format(id, pipeline),
                index_col=['gpsLongitude', 'gpsLatitude'],
                float_precision='round_trip')
        else:
            gimages = GoogleImages(data_path)
            # download the images from the relevant API
            gimages.download(GRID.lon, GRID.lat, step=config['satellite_config']['satellite_step'])
            # extract the features
            features = pd.DataFrame(gimages.featurize(GRID.lon, GRID.lat, step=config['satellite_config']['satellite_step']), index=data.index)

            features.columns = [str(col) + '_Google' for col in features.columns]
            features.to_csv(features_path)

        data = data.join(features)
        print('INFO: features extracted from Google satellite images')

    # --------------------------------- #
    # get features from Sentinel images #
    # --------------------------------- #
    if config['satellite_config']['satellite_images'] in ['Y','S']:
        features_path = "../Data/Features/features_Sentinel_id_{}_{}.csv".format(id, pipeline)
        data_path = "../Data/Satellite/"
        start_date = config["satellite_config"][0]["start_date"]
        end_date = config["satellite_config"][0]["end_date"]

        from sentinel_images import SentinelImages

        if os.path.exists(features_path):
            print('INFO: already scored.')
            features = pd.read_csv(
                features_path.format(id, pipeline),
                index_col=['gpsLongitude', 'gpsLatitude'],
                float_precision='round_trip')
        else:
            simages = SentinelImages(data_path)
            # download the images from the relevant API
            simages.download(GRID.lon, GRID.lat, start_date, end_date)
            print('INFO: scoring ...')
            # extract the features
            print('INFO: extractor instantiated.')
            features = pd.DataFrame(simages.featurize(GRID.lon, GRID.lat, start_date, end_date), index=data.index)

            features.columns = [str(col) + '_Sentinel' for col in features.columns]
            features.to_csv(features_path)

        data = data.join(features)
        print('INFO: features extracted from Sentinel images')

    # --------------- #
    # add nightlights #
    # --------------- #
    from nightlights import Nightlights

    nlights = Nightlights('../Data/Geofiles/')
    nlights.download(area, config['nightlights_date']['start'], config['nightlights_date']['end'])
    features = pd.DataFrame(nlights.featurize(GRID.lon, GRID.lat), columns=['nightlights'], index=data.index)
    # quantize nightlights
    features['nightlights'] = pd.qcut(features['nightlights'], 5, labels=False, duplicates='drop')
    data = data.join(features)

    # ---------------- #
    # add OSM features #
    # ---------------- #
    OSM = OSM_extractor(minlon, minlat, maxlon, maxlat)
    tags = {"amenity": ["school", "hospital"], "natural": ["tree"]}
    osm_gdf = {}

    for key, values in tags.items():
        for value in values:
            osm_gdf["value"] = OSM.download(key, value)
            dist = OSM.distance_to_nearest(GRID.lat, GRID.lon, osm_gdf["value"])
            data['distance_{}'.format(value)] = [np.log(0.0001 + x) for x in dist]

    # ---------------- #
    # NDBI, NDVI, NDWI #
    # ---------------- #
    print('INFO: getting NDBI, NDVI, NDWI ...')
    from rms_indexes import S2indexes

    S2 = S2indexes(area, '../Data/Geofiles/NDs/', config['NDs_date']['start'], config['NDs_date']['end'], config['scope'])
    S2.download()
    data['max_NDVI'], data['max_NDBI'], data['max_NDWI'] = S2.rms_values(GRID.lon, GRID.lat)

    # --------------- #
    # add ACLED #
    # --------------- #
    from acled import ACLED

    acled = ACLED("../Data/Geofiles/ACLED/")
    acled.download(config['iso3'], config['nightlights_date']['start'], config['nightlights_date']['end'])
    d = {}
    for property in ["fatalities", "n_events", "violence_civ"]:
        for k in [10000, 100000]:
            d[property + "_" + str(k)] = acled.featurize(GRID.lon, GRID.lat, property=property, function='density', buffer=k)

    d["weighted_sum_fatalities_by_dist"] = acled.featurize(GRID.lon, GRID.lat, property="fatalities", function='weighted_kNN')
    d["distance_to_acled_event"] = acled.featurize(GRID.lon, GRID.lat, function='distance')
    # quantize ACLED
    for c in d.keys():
        d[c] = np.nan_to_num(pd.qcut(d[c], 5, labels=False, duplicates='drop'))

    features = pd.DataFrame(d, index=data.index)
    data = data.join(features)

    # --------------- #
    # save features   #
    # --------------- #
    # drop columns with only 1 value
    print('INFO: {} columns. Dropping features with unique values (if any) ...'.format(len(data.columns)))
    data = data[[col for col in data if not data[col].nunique() == 1]]
    print('INFO: {} columns.'.format(len(data.columns)))
    # features to be use in the linear model
    features_list = list(sorted(set(data.columns) - set(['i', 'j', config['indicator']])))

    #Save non-scaled features
    data.to_csv("../Data/Features/features_all_id_{}_evaluation_nonscaled.csv".format(id))

    # Scale Features
    print("Normalizing : max")
    data[features_list] = (data[features_list] - data[features_list].mean()) / (data[features_list].max()+0.001)
    data.to_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(id))

    # --------------- #
    # model indicator #
    # --------------- #
    # shuffle dataset
    data = data.sample(frac=1, random_state=1783)  # shuffle data
    scores_dict = {} # placeholder to save the scores
    from modeller import Modeller
    X, y = data[features_list].reset_index(), data[config['indicator']]
    modeller = Modeller(X, rs_features=features_list, spatial_features=["gpsLatitude", "gpsLongitude"], scoring='r2', cv_loops=20)

    kNN_pipeline = modeller.make_model_pipeline('kNN')
    kNN_scores = modeller.compute_scores(kNN_pipeline, y)
    scores_dict['kNN_R2_mean'] = round(kNN_scores.mean(), 2)
    scores_dict['kNN_R2_std'] = round(kNN_scores.std(), 2)
    print("kNN_R2_mean: ", scores_dict['kNN_R2_mean'], "kNN_R2_std: ", scores_dict['kNN_R2_std'])

    Ridge_pipeline = modeller.make_model_pipeline('Ridge')
    Ridge_scores = modeller.compute_scores(Ridge_pipeline, y)
    scores_dict['ridge_R2_mean'] = round(Ridge_scores.mean(), 2)
    scores_dict['ridge_R2_std'] = round(Ridge_scores.std(), 2)
    print("Ridge_R2_mean: ", scores_dict['ridge_R2_mean'], "Ridge_R2_std: ", scores_dict['ridge_R2_std'])

    Ensemble_pipeline = modeller.make_ensemble_pipeline([kNN_pipeline, Ridge_pipeline])
    Ensemble_scores = modeller.compute_scores(Ensemble_pipeline, y)
    scores_dict['ensemble_R2_mean'] = round(Ensemble_scores.mean(), 2)
    scores_dict['ensemble_R2_std'] = round(Ensemble_scores.std(), 2)
    print("Ensemble_R2_mean: ", scores_dict['ensemble_R2_mean'], "Ensemble_R2_std: ", scores_dict['ensemble_R2_std'])

    # save results
    if id is None:
        write_scores_to_file(scores_dict, config['id'])
    else:
        write_scores_to_db(scores_dict, config['id'], engine)

    # ------------------------- #
    # write predictions to file #
    # ------------------------- #
    print('INFO: writing predictions to disk ...')

    from sklearn.model_selection import cross_val_predict
    results = pd.DataFrame({
        'yhat': cross_val_predict(Ensemble_pipeline, X.values, y),
        'y': data[config['indicator']].values},
        index=data.index)
    results.to_csv('../Data/Results/config_{}.csv'.format(config['id']))

    # save model for production
    Ensemble_pipeline.fit(X.values, y)

    # Best n_neighbors (kNN)
    print('INFO: number of neighbours chosen: ', Ensemble_pipeline.regr_[0].named_steps['gridsearchcv'].best_params_)
    # Best alpha (Ridge)
    print('INFO: regularization param chosen: ', Ensemble_pipeline.regr_[1].named_steps['gridsearchcv'].best_params_)

    from sklearn.externals import joblib
    joblib.dump(Ensemble_pipeline, '../Models/Ensemble_model_config_id_{}.pkl'.format(id))
    print(str(np.datetime64('now')), 'INFO: model saved.')


if __name__ == "__main__":
    print("we are in \n:", os.getcwd())
    print("and we have \n:", os.listdir("/app"))
    print("and above \n:", os.listdir("../"))
    print("and in data \n:", os.listdir("/app/Data/"))

    import tensorflow as tf
    for id in sys.argv[1:]:
        if id.isdigit():
            # use database for configs
            run(file=False, id=id)
        else:
            # use yml for configs
            print(os.getcwd())
            run(file=id)

    # rubbish collection
    tf.keras.backend.clear_session()
