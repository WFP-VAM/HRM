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
from sqlalchemy import create_engine
import yaml
import pandas as pd
from nn_extractor import NNExtractor
import numpy as np

def run():
    # ----------------- #
    # SETUP #############
    # ----------------- #
    print("INFO: config id =", sys.argv[1])

    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                    private_config['DB']['host'], private_config['DB']['database']))


    config = pd.read_sql_query("select * from config where id = {}".format(sys.argv[1]), engine)

    image_dir=os.path.join("../Data", "Satellite", config["satellite_source"][0])
    print(image_dir)
    raster=config["satellite_grid"][0]
    print(raster)

    GRID = RasterGrid(raster,image_dir)

    list_i, list_j = GRID.get_gridcoordinates(file=config["dataset_filename"][0])

    # # ----------------- #
    # # DOWNLOADING #######
    # # ----------------- #
    print("INFO: downlaoding images ...")
    GRID.download_images(list_i, list_j)
    print("INFO: images downloaded")
    # ----------------- #
    # SCORING ###########
    # ----------------- #
    from utils import scoring_postprocess
    network = NNExtractor(config['satellite_image_dir'][0], config['network_model'][0])
    print("INFO: initiating network ...")
    features = network.extract_features()
    print("INFO: extracting features")
    features = scoring_postprocess(features)
    # write out
    features.to_csv("../Data/Features/config_id_{}.csv".format(config['id'][0]), index=False)
    features=pd.read_csv("../Data/Features/config_id_{}.csv".format(config['id'][0]))


    # ----------------- #
    # ADD SURVEY DATA ###
    # ----------------- #
    hh_data = pd.read_csv(config["dataset_filename"][0])
    data = hh_data.merge(features, on=["i", "j"])
    data = data.sample(frac=1, random_state=1783).reset_index(drop=True) # shuffle data
    data_features = data[list(set(data.columns) - set(hh_data.columns) - set(['index']))]  # take only the CNN features

    # ----------------- #
    # MODEL #############
    # ----------------- #
    from evaluation_utils import MAPE, r2_pearson
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

    #data = data.loc[config['indicator'][0] > 0, :]
    data = data.loc[data[config['indicator'][0]] > 0]
    y = data[config['indicator'][0]].values  # Average normalized consumption per cluster

    if config['indicator_log'][0] == True:
        y = np.log(y)  # Log-normal distribution

    # PCA
    if config['model_pca'][0] > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=config['model_pca'][0])
        X = pca.fit_transform(data_features)
    else:
        X = data_features


    # TRAIN
    print("INFO: training model ...")
    outer_cv = KFold(5, shuffle=True, random_state=75788)
    model = Ridge()
    inner_cv = KFold(5, shuffle=True, random_state=1673)
    clf_r2 = GridSearchCV(estimator=model, param_grid=config['model_grid_parameters'][0], cv=inner_cv, scoring=r2_pearson)
    clf_MAPE = GridSearchCV(estimator=model, param_grid=config['model_grid_parameters'][0], cv=inner_cv, scoring=MAPE)
    score_r2 = cross_val_score(clf_r2, X, y, scoring=r2_pearson, cv=outer_cv)
    score_MAPE = cross_val_score(clf_MAPE, X, y, scoring=MAPE, cv=outer_cv)

    score_r2_mean, score_r2_var, score_MAPE, score_MAPE_var = score_r2.mean(), score_r2.std() * 2, score_MAPE.mean(), score_MAPE.std() * 2

    # ----------------- #
    # WRITE RESULTS PUT #
    # ----------------- #
    query = """
    insert into results (run_date, config_id, r2pearson, r2pearson_var, mape, mape_var)
    values (current_date, {}, {}, {}, {},{}) """.format(
        config['id'][0], score_r2_mean, score_r2_var, score_MAPE, score_MAPE_var)
    engine.execute(query)

if __name__ == "__main__":
    run()
