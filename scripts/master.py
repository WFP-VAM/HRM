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
sys.path.append(os.path.join("..","Src"))
from img_lib import RasterGrid
from nn_extractor import NNExtractor


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
    nightlights_date = config.get("nightlights_date")[0]
    if config['satellite_config'][0].get('satellite_images') == 'Y':
        step = config['satellite_config'][0].get("satellite_step")


    # -------- #
    # DATAPREP #
    # -------- #
    data = pd.read_csv(dataset)
    data_cols = data.columns.values

    # grid
    GRID = RasterGrid(raster)
    list_i, list_j = GRID.get_gridcoordinates(data)
    data["i"], data["j"] = list_i, list_j

    # --------------------------- #
    # GROUP CLUSTERS IN SAME TILE #
    # --------------------------- #
    # TODO: looks like shit
    cluster_N = 'countbyEA'
    print("Number of clusters: {} ".format(len(data)))

    def wavg(g, df, weight_series):
        w = df.ix[g.index][weight_series]
        return (g * w).sum() / w.sum()

    fnc = functools.partial(wavg, df=data, weight_series=cluster_N)

    try:
        data = data.groupby(["i", "j"]).agg({indicator: fnc, 'gpsLatitude': fnc, 'gpsLongitude': fnc}).reset_index()
    except KeyError:
        print("No weights, taking the average per i and j")
        data = data[['i', 'j', 'n', 'gpsLatitude', 'gpsLongitude', indicator]].groupby(["i", "j"]).mean().reset_index()

    print("Number of unique tiles: {} ".format(len(data)))

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
            GRID.download_images(list_i, list_j, step, sat, start_date, end_date)
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
    from geojson import Polygon
    from nightlights import Nightlights
    from sklearn import preprocessing

    area = Polygon([[(max(data.gpsLongitude), max(data.gpsLatitude)),
                     (max(data.gpsLongitude), min(data.gpsLatitude)),
                     (min(data.gpsLongitude), min(data.gpsLatitude)),
                     (min(data.gpsLongitude), max(data.gpsLatitude))]])

    print('INFO: adding nightlights')
    NGT = Nightlights(area, '../Data/Geofiles/nightlights/', nightlights_date)
    scaler = preprocessing.StandardScaler()
    data['nightlights'] = scaler.fit_transform(NGT.nightlights_values(data).reshape(-1, 1))

    print('INFO: nightlights correlation: ', data[indicator].corr(data['nightlights']))

    data.to_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(id), index=False)
    # ---------------- #
    # add OSM features #
    # ---------------- #
    # TODO: JB

    # --------------- #
    # model indicator #
    # --------------- #
    data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  # shuffle data
    data_features = data[list(set(data.columns) - set(data_cols) - set(['i', 'j']))]  # take only the CNN features

    # if take log of indicator
    if config['log'][0]: data[indicator] = np.log(data[indicator])
    from modeller import Modeller
    md = Modeller(['kNN', 'Kriging', 'RmSense', 'Ensamble'], data_features)
    md.compute(data[['i', 'j']], data[indicator].values)

    # save model for production
    md.save_models(id)
    print(str(np.datetime64('now')), 'INFO: model saved.')

    # ------------------ #
    # write scores to DB #
    # ------------------ #
    def default_dict_ops(d):
        r = []
        for s in d.values():
            r.append(s)
        return np.mean(r), np.var(r)

    r2, r2_var =default_dict_ops(md.scores['Ensamble'])
    r2_knn, r2_var_knn = default_dict_ops(md.scores['kNN'])
    r2_rmsense, r2_var_rmsense = default_dict_ops(md.scores['RmSense'])
    mape_rmsense = np.mean(np.abs([item for sublist in md.results['RmSense'] for item in sublist] - data[indicator])/ data[indicator])
    if mape_rmsense == float("inf") or mape_rmsense == float("-inf"): mape_rmsense = 0

    query = """
    insert into results_new (run_date, config_id, r2, r2_var, r2_knn, r2_var_knn, r2_rmsense, r2_var_rmsense, mape_rmsense)
    values (current_date, {}, {}, {}, {}, {}, {}, {}, {}) """.format(
        config['id'][0],
        r2, r2_var, r2_knn, r2_var_knn, r2_rmsense, r2_var_rmsense, mape_rmsense
        )
    engine.execute(query)

    # ------------------------- #
    # write predictions to file #
    # ------------------------- #
    print('INFO: writing predictions to disk ...')
    results = pd.DataFrame({
        'yhat': [item for sublist in md.results['kNN'] for item in sublist],
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
