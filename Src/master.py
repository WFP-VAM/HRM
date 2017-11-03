"""
- loads the survey data (already preprocessed)
- donwloads the relevant satellite images
- extracts the features with a pre-trained net
- trains a regression model to predict food insecurity
"""
import os
os.chdir('scripts')
import sys
sys.path.append("..\Src")
from img_lib import RasterGrid
from sqlalchemy import create_engine
import yaml
import pandas as pd
from nn_extractor import NNExtractor
import numpy as np

# ----------------- #
# SETUP #############
# ----------------- #
with open('../private_config.yml', 'r') as cfgfile:
    private_config = yaml.load(cfgfile)

engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                       .format(private_config['DB']['user'], private_config['DB']['password'],
                                private_config['DB']['host'], private_config['DB']['database']))


config = pd.read_sql_query("select * from config where id = 1", engine)

GRID = RasterGrid(raster=config["satellite_grid"][0],
                  image_dir=os.path.join("../Data", "Satellite", config["satellite_source"][0]))

list_i, list_j = GRID.get_gridcoordinates(file=config["dataset_filename"][0])

# ----------------- #
# DOWNLOADING #######
# ----------------- #
GRID.download_images(list_i, list_j)

# ----------------- #
# SCORING ###########
# ----------------- #
from utils import scoring_postprocess
network = NNExtractor(config['satellite_image_dir'][0], config['network_model'][0])
features = network.extract_features()
features = scoring_postprocess(features)
# write out
features.to_csv("../Data/Features/config_id_{}.csv".format(config['id'][0]), index=False)

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

data = data.loc[config['indicator'][0] > 0, :]
y = data[config['indicator'][0]].values  # Average normalized consumption per cluster

if config['indicator_log'] == True:
    y = np.log(y)  # Log-normal distribution

# PCA
if config['model_pca'][0] > 0:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=config['model_pca'][0])
    X = pca.fit_transform(data_features)
else:
    X = data_features


# TRAIN
outer_cv = KFold(5, shuffle=True, random_state=75788)
model = Ridge()
inner_cv = KFold(5, shuffle=True, random_state=1673)
clf_r2 = GridSearchCV(estimator=model, param_grid=config['model_grid_parameters'][0], cv=inner_cv, scoring=r2_pearson)
clf_MAPE = GridSearchCV(estimator=model, param_grid=config['model_grid_parameters'][0], cv=inner_cv, scoring=MAPE)
score_r2 = cross_val_score(clf_r2, X, y, scoring=r2_pearson, cv=outer_cv)
score_MAPE = cross_val_score(clf_MAPE, X, y, scoring=MAPE, cv=outer_cv)

score_r2_mean, score_r2_var, score_MAPE = score_r2.mean(), score_r2.std() * 2, score_MAPE.mean()

# ----------------- #
# WRITE RESULTS PUT #
# ----------------- #
query = """
insert into results (run_date, config_id, r2pearson, r2pearson_var, mape)
values (current_date, {}, {}, {}, {}) """.format(
    config['id'][0], score_r2_mean, score_r2_var, score_MAPE)
engine.execute(query)
