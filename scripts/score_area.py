# -*- coding: utf-8 -*-
"""
- downloads the pictures relevant for scoring
- extracts features
- loads a pre-trained model
- makes predictions
- plots
"""
import os
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.externals import joblib
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..", "Src"))
from base_layer import BaseLayer
from google_images import GoogleImages
import yaml
from osm import OSM_extractor
from utils import points_to_polygon, tifgenerator, aggregate, boundaries
import rasterio
from rasterio.mask import mask
import click


# ---------- #
# PARAMETERS #
@click.command()
@click.option('--id', type=int)
@click.option('--aggregate_factor', default=1, type=int)
@click.option('--min_pop', default=0, type=float)
@click.option('--bbox', nargs=4, default=(0,0,0,0), required=False, type=float, help='bounding box <minlat> <minlon> <maxlat> <maxlon>')
@click.option('--shapefile', default=None, type=str)
def main(id, aggregate_factor, min_pop, bbox, shapefile):

    # read the configs for id
    print(str(np.datetime64('now')), " INFO: config id =", id)

    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    config = pd.read_sql_query("select * from config_new where id = {}".format(id), engine)
    dataset = config.get("dataset_filename")[0]
    raster = config["satellite_grid"][0]
    scope = config["scope"][0]
    nightlights_date_start, nightlights_date_end = config["nightlights_date"][0].get("start"), \
                                                   config["nightlights_date"][0].get("end")
    s2_date_start, s2_date_end = config["NDs_date"][0].get("start"), config["NDs_date"][0].get("end")
    ISO = config["iso3"][0]
    if config['satellite_config'][0].get('satellite_images') == 'Y':
        print('INFO: satellite images from Google and Sentinel-2')
        step = config['satellite_config'][0].get("satellite_step")
    elif config['satellite_config'][0].get('satellite_images') == 'G':
        print('INFO: only Google satellite images.')
        step = config['satellite_config'][0].get("satellite_step")
    elif config['satellite_config'][0].get('satellite_images') == 'N':
        print('INFO: no satellite images')


    # ----------------------------------- #
    # WorldPop Raster too granular (lots of images), aggregate #
    if aggregate_factor > 1:
        print('INFO: aggregating raster with factor {}'.format(aggregate_factor))
        base_raster = "../local_raster.tif"
        aggregate(raster, base_raster, aggregate_factor)
    else:
        base_raster = raster

    # ---------------- #
    # AREA OF INTEREST #
    # ---------------- #
    # dataset_df = pd.read_csv(dataset)
    # data_cols = dataset_df.columns.values

    if sum(bbox) != 0:  # dummy bbox
        print("INFO: using AOI from bbox")
        print(sum(bbox))
        # define AOI with manually defined bbox
        minlat, minlon, maxlat, maxlon = bbox[0], bbox[1], bbox[2], bbox[3]
        area = points_to_polygon(minlat=minlat, minlon=minlon, maxlat=maxlat, maxlon=maxlon)
    else:
        print("INFO: using AOI from dataset.")
        # use dataset's extent
        dataset_df = pd.read_csv(dataset)
        minlat, maxlat, minlon, maxlon = boundaries(dataset_df['gpsLatitude'], dataset_df['gpsLongitude'])
        area = points_to_polygon(minlat=minlat, minlon=minlon, maxlat=maxlat, maxlon=maxlon)
        del dataset_df

    # crop raster
    with rasterio.open(base_raster) as src:
        out_image, out_transform = mask(src, [area], crop=True)
        out_meta = src.meta.copy()

    # save the resulting raster
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform
                     })

    final_raster = "../final_raster.tif"
    print('INFO: Removing tiles with population under {}'.format(min_pop))  # only score areas where there are at agg factor living
    with rasterio.open(final_raster, "w", **out_meta) as dest:
        out_image[out_image < min_pop] = dest.nodata
        dest.write(out_image)
        list_j, list_i = np.where(out_image[0] != dest.nodata)

    # instantiate GRID
    GRID = BaseLayer(final_raster)

    coords_x, coords_y = np.round(GRID.get_gpscoordinates(list_i, list_j), 5)

    ix = pd.MultiIndex.from_arrays([list_i, list_j, coords_y, coords_x], names=('i', 'j', "gpsLatitude", "gpsLongitude"))

    print("Number of clusters: {} ".format(len(ix)))

    pipeline = 'scoring'

    # ------------------------------------------------ #
    # download images from Google and Extract Features #
    # ------------------------------------------------ #
    if config['satellite_config'][0].get('satellite_images') in ['Y', 'G']:
        features_path = "../Data/Features/features_Google_id_{}_{}.csv".format(id, pipeline)
        data_path = "../Data/Satellite/"

        gimages = GoogleImages(data_path)
        # download the images from the relevant API
        gimages.download(coords_x, coords_y, step=step)
        # extract the features
        features = pd.DataFrame(gimages.featurize(coords_x, coords_y, step=step), index=ix)
        features.columns = [str(col) + '_Google' for col in features.columns]
        features.to_csv(features_path)
        print('INFO: features extracted.')
        data = features.copy()
    # ------------------------------------------------------------- #
    # download Sentinel images and Extract Features #
    # ------------------------------------------------------------- #
    if config['satellite_config'][0].get('satellite_images') == 'Y':
        features_path = "../Data/Features/features_Sentinel_id_{}_{}.csv".format(id, pipeline)
        data_path = "../Data/Satellite/"
        start_date = config["satellite_config"][0]["start_date"]
        end_date = config["satellite_config"][0]["end_date"]

        from sentinel_images import SentinelImages

        simages = SentinelImages(data_path)
        # download the images from the relevant API
        simages.download(coords_x, coords_y, start_date, end_date)
        print('INFO: scoring ...')
        # extract the features
        print('INFO: extractor instantiated.')
        features = pd.DataFrame(simages.featurize(coords_x, coords_y, start_date, end_date), index=ix)

        features.columns = [str(col) + '_Sentinel' for col in features.columns]
        features.to_csv(features_path)

        if data is not None:
            data = data.join(features)
        else:
            data = features.copy()
        print('INFO: features extracted')

    # --------------- #
    # add nightlights #
    # --------------- #
    from nightlights import Nightlights

    nlights = Nightlights('../Data/Geofiles/')
    nlights.download(area, nightlights_date_start, nightlights_date_end)
    features = pd.DataFrame(nlights.featurize(coords_x, coords_y), columns=['nightlights'], index=ix)

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
            dist = OSM.distance_to_nearest(coords_y, coords_x, osm_gdf["value"])
            data['distance_{}'.format(value)] = [np.log(0.0001 + x) for x in dist]

    # ---------------- #
    #   NDBI,NDVI,NDWI #
    # ---------------- #
    print('INFO: getting NDBI, NDVI, NDWI ...')
    from rms_indexes import S2indexes

    S2 = S2indexes(area, '../Data/Geofiles/NDs/', s2_date_start, s2_date_end, scope)
    S2.download()
    data['max_NDVI'], data['max_NDBI'], data['max_NDWI'] = S2.rms_values(coords_x, coords_y)

    # --------------- #
    # add ACLED #
    # --------------- #
    from acled import ACLED

    acled = ACLED("../Data/Geofiles/ACLED/")
    acled.download(ISO, nightlights_date_start, nightlights_date_end)
    d = {}
    for property in ["fatalities", "n_events", "violence_civ"]:
        for k in [10000, 100000]:
            d[property + "_" + str(k)] = acled.featurize(coords_x, coords_y, property=property, function='density', buffer=k)

    d["weighted_sum_fatalities_by_dist"] = acled.featurize(coords_x, coords_y, property="fatalities", function='weighted_kNN')
    d["distance_to_acled_event"] = acled.featurize(coords_x, coords_y, function='distance')

    features = pd.DataFrame(d, index=data.index)
    data = data.join(features)

    # --------------- #
    # save features   #
    # --------------- #
    # features to be use in the linear model
    features_list = list(sorted(data.columns))
    print('features list : \n', features_list)
    # Scale Features
    print("Normalizing : max")
    data[features_list] = (data[features_list] - data[features_list].mean()) / data[features_list].max()

    data.to_csv("../Data/Features/features_all_id_{}_{}.csv".format(id, pipeline))

    # ------- #
    # predict #
    # ------- #
    ensemble_pipeline = joblib.load('../Models/Ensemble_model_config_id_{}.pkl'.format(id))
    print(str(np.datetime64('now')), 'INFO: model loaded.')

    X = data.reset_index(level=[2,3])
    ensemble_predictions = ensemble_pipeline.predict(X.values)

    # if take log of indicator
    if config['log'][0]:
        ensemble_predictions = np.exp(ensemble_predictions)

    results = pd.DataFrame({'i': list_i, 'j': list_j, 'lat': coords_y, 'lon': coords_x, 'yhat': ensemble_predictions})
    results.to_csv('../Data/Results/config_{}.csv'.format(id))
    outfile = "../Data/Results/scalerout_{}.tif".format(id)
    tifgenerator(outfile=outfile,
                 raster_path=final_raster,
                 df=results)

    outfile = "../Data/Results/scalerout_{}_kNN.tif".format(id)
    results['yhat_kNN'] = ensemble_pipeline.regr_[0].predict(X.values)
    tifgenerator(outfile=outfile, raster_path=final_raster, df=results, value='yhat_kNN')

    outfile = "../Data/Results/scalerout_{}_Ridge.tif".format(id)
    results['yhat_Ridge'] = ensemble_pipeline.regr_[1].predict(X.values)
    tifgenerator(outfile=outfile, raster_path=final_raster, df=results, value='yhat_Ridge')

    if shapefile is not None:
        input_rst = "../Data/Results/scalerout_{}.tif".format(id)
        weight_rst = "../tmp/final_raster.tif"

        output_shp = "../Data/Results/scalerout_{}_aggregated.shp".format(id)
        from utils import weighted_sum_by_polygon
        weighted_sum_by_polygon(shapefile, input_rst, weight_rst, output_shp)


if __name__ == '__main__':

    main()
