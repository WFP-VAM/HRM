"""
- loads the survey data (already preprocessed)
- donwloads the relevant satellite images
- extracts the features with a pre-trained net
- trains a regression model to predict food insecurity
"""
import os
import sys
sys.path.append(os.path.join("..","Src"))
from img_lib import RasterGrid
from img_utils import getRastervalue
from sqlalchemy import create_engine
import yaml
import pandas as pd
from nn_extractor import NNExtractor
import numpy as np
import functools


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

    config = pd.read_sql_query("select * from config_new where id = {}".format(id), engine)  # TODO: reading from config_new
    dataset = config["dataset_filename"][0]
    indicator = config["indicator"][0]
    raster = config["satellite_grid"][0]
    step = config["satellite_step"][0]
    start_date = config["sentinel_config"][0]["start_date"]
    end_date = config["sentinel_config"][0]["end_date"]
    land_use_raster = config["land_use_raster"][0]
    output = config['output'][0]
    model_grid_parameters = config['model_grid_parameters'][0]

    # -------- #
    # DATAPREP #
    # -------- #
    data = pd.read_csv(dataset)
    data_cols = data.columns.values

    # grid
    GRID = RasterGrid(raster)
    list_i, list_j = GRID.get_gridcoordinates(data)

    data["i"], data["j"] = list_i, list_j

    # Grouping clusters that belong to the same tile. # TODO: looks like shit
    cluster_N = 'countbyEA'
    print("Number of clusters: {} ".format(len(data)))

    def wavg(g, df, weight_series):
        w = df.ix[g.index][weight_series]
        return (g * w).sum() / w.sum()

    fnc = functools.partial(wavg, df=data, weight_series=cluster_N)

    try:
        data = data.groupby(["i", "j"]).agg({indicator: fnc, 'gpsLatitude': fnc, 'gpsLongitude': fnc}).reset_index()
    except:
        print("No weights, taking the average per i and j")
        data = data[['i', 'j', 'gpsLatitude', 'gpsLongitude', indicator]].groupby(["i", "j"]).mean().reset_index()

    print("Number of unique tiles: {} ".format(len(data)))

    list_i, list_j, pipeline = data["i"], data["j"], 'evaluation'

    # ---------------------------------------- #
    # download images from Google and Sentinel #
    # ---------------------------------------- #
    for sat in ['Google', 'Sentinel']:
        print('INFO: routine for provider: ', sat)
        # dopwnlaod the images from the relevant API
        GRID.download_images(list_i, list_j, step, sat, start_date, end_date)
        print('INFO: images downloaded.')

        if os.path.exists("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline)):
            print('INFO: already scored.')
            features = pd.read_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline))
        else:
            print('INFO: scoring ...')
            # extarct the features
            network = NNExtractor(id, sat, GRID.image_dir, sat, step, GRID)
            print('INFO: extractor instantiated.')

            features = network.extract_features(list_i, list_j, sat, start_date, end_date, pipeline)
            features.to_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline), index=False)

        features = features.drop('index', 1)
        data = data.merge(features, on=["i", "j"])

    data.to_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(id), index=False)

    print('INFO: features extracted.')

    # ----------- #
    # add landuse #
    # ----------- #
    if land_use_raster is not None:
        print("INFO: adding land use.")
        data["land_use"] = getRastervalue(data, land_use_raster)

    # --------------- #
    # model indicator #
    # --------------- #
    data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  # shuffle data
    data_features = data[list(set(data.columns) - set(data_cols) - set(['i', 'j']))]  # take only the CNN features

    from modeller import Modeller
    md = Modeller(['kNN', 'Kriging', 'RmSense'], data_features)
    md.compute(data[['i', 'j']], data[indicator].values)

    # save model for production
    md.save_models(id)
    print(str(np.datetime64('now')), 'INFO: model saved.')

    # ------------------ #
    # write scores to DB #
    # ------------------ #
    query = """
    insert into results_new (run_date, config_id, r2, r2_var, r2_knn, r2_var_knn, r2_rmsense, r2_var_rmsense)
    values (current_date, {}, {}, {}, {}, {}, {}, {}) """.format(
        config['id'][0],
        md.scores['combined'], md.vars['combined'],
        md.scores['kNN'], md.vars['kNN'],
        md.scores['RmSense'], md.vars['RmSense'])
    engine.execute(query)

    # ------------------------- #
    # write predictions to file #
    # ------------------------- #
    print('INFO: writing predictions to disk ...')
    results = pd.DataFrame({
        'yhat': (md.RmSense.predict(data_features) + md.kNN.predict(data[['i', 'j']])) / 2.,
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
