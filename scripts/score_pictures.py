"""
- downloads the pictures relevant for scoring
- extracts features
- loads a pre-trained model
- makes predictions
"""
import os
os.chdir('scripts')
import sys
sys.path.append(os.path.join("..", "Src"))
from img_lib import RasterGrid
from sqlalchemy import create_engine
import yaml
import pandas as pd
from nn_extractor import NNExtractor
from utils import scoring_postprocess, get_coordinates_of_country

with open('../private_config.yml', 'r') as cfgfile:
    private_config = yaml.load(cfgfile)

engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                       .format(private_config['DB']['user'], private_config['DB']['password'],
                               private_config['DB']['host'], private_config['DB']['database']))

config = pd.read_sql_query("select * from config where id = {}".format(1), engine)
image_dir = os.path.join("../Data", "Satellite", config["satellite_source"][0])

# get raster
GRID = RasterGrid(config["satellite_grid"][0], image_dir)

# get all coordinates in Uganda
uga_cords = get_coordinates_of_country("../Data/Shapefiles/UGA_adm_shp/UGA_adm0.shp", spacing=0.5)

list_i, list_j = GRID.get_gridcoordinates(file="../Data/datasets/Uganda_to_score.csv")
GRID.download_images(list_i, list_j, step=0)

# extract features
network = NNExtractor("""..\Data\Satellite\\test""",
                      config['network_model'][0], step=0)
features = network.extract_features()
features = scoring_postprocess(features)

# load model
from sklearn.externals import joblib
clf = joblib.load('../Model/.pkl')