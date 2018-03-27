
def download(id, data, GRID, list_i, list_j, raster, step, sat, start_date, end_date):
    import os

    print("Starting pipeline for: {}".format(sat))

    if raster == "../Data/Satellite/F182013.v4c_web.stable_lights.avg_vis.tif":
        image_dir = os.path.join("../Data", "Satellite", sat)
    else:
        image_dir = os.path.join("../Data", "Satellite", sat, os.path.splitext(os.path.basename(raster))[0])
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    GRID.output_image_dir = image_dir + "/"

    print("Images directory: {}".format(GRID.output_image_dir))

    GRID.download_images(list_i, list_j, step, sat, start_date, end_date)


def score_merge(id, data, GRID, list_i, list_j, raster, step, sat, start_date, end_date, network_model, custom_weights, pipeline="evaluation"):
    import os
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append(os.path.join(".." "Src"))
    from nn_extractor import NNExtractor

    print(str(np.datetime64('now')), " INFO: initiating network ...")

    image_dir = GRID.output_image_dir

    network = NNExtractor(id, sat, image_dir, network_model, step)
    if custom_weights is not None:
        network.load_weights(custom_weights)

    if os.path.exists("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline)):
        features = pd.read_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline))
    else:
        features = network.extract_features(list_i, list_j, sat, start_date, end_date, pipeline)
        features.to_csv("../Data/Features/features_{}_id_{}_{}.csv".format(sat, id, pipeline), index=False)

    try:
        features = features.drop('index', 1)
    except:
        pass

    data = data.merge(features, on=["i", "j"])

    return data