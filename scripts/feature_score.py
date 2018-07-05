import pandas as pd
import click
import sys
import os
import yaml
from sqlalchemy import create_engine
import numpy as np

try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..", "Src"))

from evaluation_utils import r2_pearson


@click.command()
@click.option('--id')
@click.option('--indicator', default=(None))
@click.option('--scoring', default=('r2'))
@click.option('--cv', default=(10))
def individual_r2(id, indicator, scoring, cv):
    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    config = pd.read_sql_query("select * from config_new where id = {}".format(id), engine)
    dataset = config.get("dataset_filename")[0]
    if indicator is None:
        indicator = config["indicator"][0]
    data = pd.read_csv(dataset)
    data_cols = data.columns.values
    print(data_cols)

    data = pd.read_csv("../Data/Features/features_all_id_{}_evaluation.csv".format(id))

    data["noise"] = np.random.normal(0, 1, len(data))

    data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  # shuffle data

    features_list = list(set(data.columns) - set(data_cols) - set(['i', 'j', 'gpsLatitude', 'gpsLongitude', 'cluster', 'n', indicator, "log_".format(indicator)]))
    print(features_list)
    nn_features_google = [i for i in features_list if i.endswith('_Google')]
    nn_features_sentinel = [i for i in features_list if i.endswith('_Sentinel')]
    nn_features = nn_features_google + nn_features_sentinel
    print(nn_features)
    no_nn_features = list(set(features_list) - set(nn_features))
    print(no_nn_features)

    # if take log of indicator
    if config['log'][0]:
        data[indicator] = np.log(data[indicator])

    if scoring == 'r2_pearson':
        scoring = r2_pearson

    X = data
    print("indicator: ", indicator)
    y = data[indicator]
    cv_loops = cv
    from modeller import Modeller
    Modeller_all = Modeller(X, rs_features=features_list, spatial_features=["gpsLatitude", "gpsLongitude"], scoring=scoring, cv_loops=cv_loops)

    kNN_pipeline = Modeller_all.make_model_pipeline('kNN')
    kNN_scores = Modeller_all.compute_scores(kNN_pipeline, y)
    kNN_R2_mean = kNN_scores.mean()
    kNN_R2_std = kNN_scores.std()
    print("kNN_R2_mean: ", round(kNN_R2_mean, 2), "kNN_R2_std: ", round(kNN_R2_std, 2))

    Ridge_pipeline = Modeller_all.make_model_pipeline('Ridge')
    Ridge_scores = Modeller_all.compute_scores(Ridge_pipeline, y)
    Ridge_R2_mean = Ridge_scores.mean()
    Ridge_R2_std = Ridge_scores.std()
    print("Ridge_R2_mean: ", round(Ridge_R2_mean, 2), "Ridge_R2_std: ", round(Ridge_R2_std, 2))

    Ensemble_pipeline = Modeller_all.make_ensemble_pipeline([kNN_pipeline, Ridge_pipeline])
    Ensemble_scores = Modeller_all.compute_scores(Ensemble_pipeline, y)
    Ensemble_R2_mean = Ensemble_scores.mean()
    Ensemble_R2_std = Ensemble_scores.std()
    print("Ensemble_R2_mean: ", round(Ensemble_R2_mean, 2), "Ensemble_R2_std: ", round(Ensemble_R2_std, 2))

    Modeller_google = Modeller(X, rs_features=nn_features_google, spatial_features=["gpsLatitude", "gpsLongitude"], scoring=scoring, cv_loops=cv_loops)
    Ridge_pipeline = Modeller_google.make_model_pipeline('Ridge')
    Ridge_scores_google = Modeller_google.compute_scores(Ridge_pipeline, y)
    Ridge_R2_mean_google = Ridge_scores_google.mean()
    Ridge_R2_std_google = Ridge_scores_google.std()
    print("Ridge_R2_google_mean: ", round(Ridge_R2_mean_google, 2), "Ridge_R2_google_std: ", round(Ridge_R2_std_google, 2))

    Modeller_sentinel = Modeller(X, rs_features=nn_features_sentinel, spatial_features=["gpsLatitude", "gpsLongitude"], scoring=scoring, cv_loops=cv_loops)
    Ridge_pipeline = Modeller_sentinel.make_model_pipeline('Ridge')
    Ridge_scores_sentinel = Modeller_sentinel.compute_scores(Ridge_pipeline, y)
    Ridge_R2_mean_sentinel = Ridge_scores_sentinel.mean()
    Ridge_R2_std_sentinel = Ridge_scores_sentinel.std()
    print("Ridge_R2_sentinel_mean: ", round(Ridge_R2_mean_sentinel, 2), "Ridge_R2_sentinel_std: ", round(Ridge_R2_std_sentinel, 2))

    Modeller_nn = Modeller(X, rs_features=nn_features, spatial_features=["gpsLatitude", "gpsLongitude"], scoring=scoring, cv_loops=cv_loops)
    Ridge_pipeline = Modeller_nn.make_model_pipeline('Ridge')
    Ridge_scores_nn = Modeller_nn.compute_scores(Ridge_pipeline, y)
    Ridge_R2_mean_nn = Ridge_scores_nn.mean()
    Ridge_R2_std_nn = Ridge_scores_nn.std()
    print("Ridge_R2_nn_mean: ", round(Ridge_R2_mean_nn, 2), "Ridge_R2_nn_std: ", round(Ridge_R2_std_nn, 2))

    Modeller_no_nn = Modeller(X, rs_features=no_nn_features, spatial_features=["gpsLatitude", "gpsLongitude"], scoring=scoring, cv_loops=cv_loops)
    Ridge_pipeline = Modeller_no_nn.make_model_pipeline('Ridge')
    Ridge_scores_no_nn = Modeller_no_nn.compute_scores(Ridge_pipeline, y)
    Ridge_R2_mean_no_nn = Ridge_scores_no_nn.mean()
    Ridge_R2_std_no_nn = Ridge_scores_no_nn.std()
    print("Ridge_R2_no_nn_mean: ", round(Ridge_R2_mean_no_nn, 2), "Ridge_R2_no_nn_std: ", round(Ridge_R2_std_no_nn, 2))


    for feature in no_nn_features:
        Modeller_feature = Modeller(X, rs_features=feature, scoring=scoring, cv_loops=cv_loops)
        Ridge_pipeline = Modeller_feature.make_model_pipeline('Ridge')
        Ridge_scores_feature = Modeller_feature.compute_scores(Ridge_pipeline, y)
        Ridge_R2_mean_feature = Ridge_scores_feature.mean()
        Ridge_R2_std_feature = Ridge_scores_feature.std()

        all_but_feature = list(set(features_list) - set([feature]))
        Modeller_all_but_feature = Modeller(X, rs_features=all_but_feature, scoring=scoring, cv_loops=cv_loops)
        Ridge_pipeline2 = Modeller_all_but_feature.make_model_pipeline('Ridge')
        Ridge_scores_all_but_feature = Modeller_all_but_feature.compute_scores(Ridge_pipeline2, y)
        Ridge_R2_mean_all_but_feature = Ridge_scores_all_but_feature.mean()
        print(feature, " R2_mean: ", round(Ridge_R2_mean_feature, 2), " R2_mean_added_value: ", round(Ridge_R2_mean - Ridge_R2_mean_all_but_feature, 2), "R2_std: ", round(Ridge_R2_std_feature, 2))


if __name__ == "__main__":
    individual_r2()
