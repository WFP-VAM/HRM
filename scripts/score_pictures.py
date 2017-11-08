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
from utils import scoring_postprocess, get_coordinates_from_shp

# --------------- #
# SETUP ###########
# load config
with open('../private_config.yml', 'r') as cfgfile:
    private_config = yaml.load(cfgfile)

# connect to db and read config table
engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                       .format(private_config['DB']['user'], private_config['DB']['password'],
                               private_config['DB']['host'], private_config['DB']['database']))

config = pd.read_sql_query("select * from config where id = {}".format(1), engine)
# TODO
image_dir = os.path.join("../Data", "Satellite", "test")

# get raster
GRID = RasterGrid(config["satellite_grid"][0], image_dir)

# ---------------------- #
# IMAGES IN SCOPE ########
# ADM0
adm0_lat, adm0_lon = get_coordinates_from_shp("../Data/Shapefiles/UGA_adm_shp/UGA_adm0.shp", spacing=0.5)
# get the matching tiles
list_i, list_j = GRID.get_gridcoordinates2(adm0_lat, adm0_lon)
# download images to predict
GRID.download_images(list_i, list_j, step=0, provider='Google')
# extract features
network = NNExtractor("""..\Data\Satellite\\test""", config['network_model'][0], step=0)
features = scoring_postprocess(network.extract_features())
X = features.drop(['index', 'i', 'j'], axis=1)

# load model and predict
from sklearn.externals import joblib
clf = joblib.load('../Models/ridge_model_config_id_1.pkl')
yhat = clf.predict(X)

# make simple plot
import folium
m = folium.Map(
    location=[1.130956, 32.354771],
    tiles='Stamen Terrain',
    zoom_start=6
)

for i in range(0, len(yhat)-1):
    folium.CircleMarker(
        location=[uga_lat[i], uga_lon[i]],
        radius=yhat[i]*10,
        popup=str(yhat[i]),
        color='#3186cc',
        fill=True,
        fill_color='#3186cc'
    ).add_to(m)
m.save('map.html')