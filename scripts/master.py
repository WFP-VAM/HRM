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
from utils import scoring_postprocess


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

    print(provider)

    GRID = RasterGrid(raster)
    list_i, list_j = GRID.get_gridcoordinates(dataset)#

    hh_data = pd.read_csv(dataset)

    data=hh_data
    data["i"] = list_i
    data["j"] = list_j

    #data=data.groupby(["i","j"], as_index=False).apply(lambda x: np.average(x, weights=x['countbyEA']))

    for sat in provider.split(","):

        print(sat)
        image_dir = os.path.join("../Data", "Satellite", sat)
        GRID.output_image_dir = image_dir + "/"

        # # ----------------- #
        # # DOWNLOADING #######
        # # ----------------- #

        GRID.download_images(list_i, list_j, step, sat, start_date, end_date)


        # # ----------------- #
        # # SCORING #######
        # # ----------------- #

        print(str(np.datetime64('now')), " INFO: initiating network ...")
        network = NNExtractor(id, sat, image_dir, network_model, step)
        if custom_weights is not None:
            network.load_weights(custom_weights)
        features = network.extract_features(list_i, list_j, sat, start_date, end_date)
        features.to_csv("../Data/Features/features_{}_config_id_{}.csv".format(sat,id), index=False)

        # # ----------------- #
        # # ADD SURVEY DATA #######
        # # ----------------- #

        data = data.merge(features, on=["i", "j"])
        data.to_csv("../Data/Features/features_all_config_id_{}.csv".format(id), index=False)

    # ----------------- #
    # ADD OTHER FEATURES  ###
    # ----------------- #
    if land_use_raster is not None:
        raster_file = land_use_raster
        data["land_use"] = data.apply(getRastervalue, args=(raster_file,), axis=1)

    data = data.loc[data[indicator] > 0]
    data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  #shuffle data
    data_features = data[list(set(data.columns) - set(hh_data.columns) - set(['index','index_x','index_y']))]  # take only the CNN features

    # ----------------- #
    # MODEL #############
    # ----------------- #
    from evaluation_utils import MAPE, r2_pearson, r2
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict

    y = data[indicator].values

    # Log-normal distribution
    if indicator_log == True:
        y = np.log(y)

    # PCA
    if model_pca > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=model_pca)
        X = pca.fit_transform(data_features)
    else:
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
