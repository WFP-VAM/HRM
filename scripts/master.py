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

    # dataset
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

    # download images from Google and Sentinel
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

    # ----------------- #
    # ADD OTHER FEATURES  ###
    # ----------------- #
    if land_use_raster is not None:
        print("INFO: adding land use.")
        data["land_use"] = getRastervalue(data, land_use_raster)

    # ----------------- #
    # MODEL #############
    # ----------------- #
    data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  # shuffle data
    data_features = data[list(set(data.columns) - set(data_cols) - set(['i', 'j']))]  # take only the CNN features

    from evaluation_utils import MAPE, r2_pearson, r2
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict

    y = data[indicator].values
    X = data_features

    # TRAIN MODEL
    outer_cv = KFold(5, shuffle=True, random_state=75788)
    inner_cv = KFold(5, shuffle=True, random_state=1673)
    print(str(np.datetime64('now')), " INFO: training model ...")
    if output == 'regression':
        from sklearn.linear_model import Ridge
        model = Ridge()
        clf = GridSearchCV(estimator=model, param_grid=model_grid_parameters, cv=inner_cv, scoring=r2_pearson)

        score = cross_val_score(clf, X, y, scoring=r2_pearson, cv=outer_cv)
        score_r2 = cross_val_score(clf, X, y, scoring=r2, cv=outer_cv)
        score_MAPE = cross_val_score(clf, X, y, scoring=MAPE, cv=outer_cv)

        predict = cross_val_predict(clf, X, y, cv=outer_cv)

    elif output == 'classification':
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier()
        clf = GridSearchCV(estimator=model, param_grid=model_grid_parameters, cv=inner_cv)

        score = cross_val_score(clf, X, y, cv=outer_cv)
        predict = cross_val_predict(clf, X, y, cv=outer_cv)

    print('INFO: Pearson score: ', score.mean())
    # WRITE FULL RESULTS to FILE SYSTEM
    if output == 'regression': results_df = pd.DataFrame([predict, y],
                                                                      index=["predict", "y"]).T
    if output == 'classification': results_df = pd.DataFrame([predict, y],
                                                                      index=["predict", "y"]).T

    # attach coordinates
    results_df['i'], results_df['j'] = data.i, data.j

    if not os.path.exists('../Data/Results'):
        os.makedirs('../Data/Results')
    results_df.to_csv(os.path.join("../Data/Results", "confi_"+str(id)+"_results.csv"), index=False)

    # SAVE MODEL FOR PRODUCTION
    from sklearn.externals import joblib
    prod_cv = clf.fit(X, y)
    print('INFO: best parameter: ', prod_cv.best_params_)
    model_prod = Ridge(alpha=prod_cv.best_params_['alpha'])
    model_prod.fit(X, y)
    if not os.path.exists('../Models'):
        os.makedirs('../Models')
    joblib.dump(model_prod, '../Models/ridge_model_config_id_{}.pkl'.format(id))
    print(str(np.datetime64('now')), 'INFO: model saved.')

    # ------------------ #
    # WRITE SCORES to DB #
    # ------------------ #
    if output == 'regression':
        score_mean, score_var, score_r2_mean, score_r2_var, score_MAPE, score_MAPE_var = \
        score.mean(), score.std() * 2, score_r2.mean(), score_r2.std() * 2, score_MAPE.mean(), score_MAPE.std() * 2

        query = """
        insert into results (run_date, config_id, r2, r2_var, r2pearson, r2pearson_var, mape, mape_var)
        values (current_date, {}, {}, {}, {}, {}, {},{}) """.format(
            config['id'][0], score_r2_mean, score_r2_var, score_mean, score_var, score_MAPE, score_MAPE_var)
        engine.execute(query)

    if output == 'classification':
        score, score_mean = score.mean(), score.std() * 2

        query = """
                insert into results (run_date, config_id, r2, r2_var)
                values (current_date, {}, {}, {}) """.format(
                id, score, score_mean)
        engine.execute(query)


if __name__ == "__main__":

    import tensorflow as tf
    for id in sys.argv[1:]:
        run(id)

    # rubbish collection
    tf.keras.backend.clear_session()
