# ----------------------------------------------------------------------
# script to assess the features extracted by the network over the images
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# libraries
import os
os.chdir("""scripts""")
import pandas as pd
import sys
sys.path.append("../Src")
from img_lib import RasterGrid
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# import and process the features created

features = pd.read_csv('../Data/Features/features_config_id_134.csv')

non_feature_columns = ['index', 'i', 'j']
feature_columns = list(set(features.columns.values) - set(non_feature_columns))
feature_matrix = features[feature_columns]
feature_matrix = feature_matrix.reindex_axis(sorted(feature_matrix.columns), axis=1)


# ----------------------------------------------------------------------
# apply PCA
from sklearn.decomposition import PCA
pc = PCA(n_components=2)
x = pc.fit_transform(features[feature_columns])
dfx = pd.DataFrame(x, columns=['x', 'y'])


# ----------------------------------------------------------------------
# get raster and retrieve relevant raster coordinates
GRId = RasterGrid(raster='../Data/Satellite/F182013.v4c_web.stable_lights.avg_vis.tif', image_dir='../Data/Satellite/Google')
dfx['i'], dfx['j'] = features['i'], features['j']

# for what clusters?
dfx['lon'], dfx['lat'] = GRId.get_gpscoordinates(dfx['i'], dfx['j'], step=0)
dfx['lonlat'] = dfx[['lon', 'lat']].round(4).astype(str).apply(lambda x: ','.join(x), axis = 1)

# ----------------------------------------------------------------------
# add indicators score
hh_data = pd.read_csv("../Data/datasets/VAM_ENSA_Nigeria_national_2017_indiv_reduced.csv")[['FCS', 'i', 'j']]

dfx = pd.merge(dfx, hh_data, on=['i', 'j'])

# ----------------------------------------------------------------------
# export for Viz
dfx.to_clipboard()


# ----------------------------------------------------------------------
# attach images
from PIL import Image

# output_image_dir=os.path.join("../Data","Satellite", GRId.config["satellite"]["source"])
#
# for dir in os.listdir(output_image_dir):
#     image_dir = os.path.join(output_image_dir, dir)
#     for name in os.listdir(image_dir):
#
#         if ((name.endswith(".jpg")) & (name == dir+'.jpg')):

            #image = Image.open(output_image_dir+ "\\" + dir + "\\" +name)
            #dfx.loc[((int(dfx['i'][0])) & (dfx.j == int(name[6:10][0]))), 'image'] = image




#-----------------------------------------------------------------------
# matplotlib viz
plt.subplots_adjust(bottom=0.1)
plt.scatter(
    dfx['x'], dfx['y'], marker='o', c=dfx['i'], s=dfx['j']/1000,
    cmap=plt.get_cmap('Spectral'))

for label, x, y in zip(dfx['lonlat'], dfx['x'], dfx['y']):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-10, 10),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


#-----------------------------------------------------------------------
# matplotlib viz for colour

pd.cut(np.log(dfx.cons), 4).unique()
dfx['cons_cat'] = pd.cut(np.log(dfx.cons), 4, labels=False)
dfx['cons_cat'].replace({0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}, inplace=True)
fig, ax = plt.subplots()
for cat in dfx.cons_cat.unique():
    print(cat)
    ax.scatter(
        dfx.loc[dfx['cons_cat'] == cat, 'x'],
        dfx.loc[dfx['cons_cat'] == cat, 'y'], c=cat, label=cat)
ax.legend()
plt.show()

# ----------------------------------------------------------------------
# other
# # plotly style
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.tools as pltl
# pltl.set_credentials_file(username='lo.riches', api_key='FloJa3YqhrQCGsHsTMLl')
# # Create a trace
# trace_lon = go.Scatter(
#     x=dfx.x,
#     y=dfx.y,
#     mode='markers',
#     marker=dict(
#             size='12',
#             color = dfx.lon, #set color equal to a variable
#             colorscale='Viridis',
#             showscale=True
#         )
# )
# trace_lat = go.Scatter(
#     x=dfx.x,
#     y=dfx.y,
#     mode='markers',
#     marker=dict(
#             size='12',
#             color = dfx.lat, #set color equal to a variable
#             colorscale='Viridis',
#             showscale=True
#         )
# )
# layout = go.Layout(title='PCA of the features', width=800, height=640)
# fig = pltl.make_subplots(rows=1, cols=2)
# fig.append_trace(trace_lon, 1, 1)
# fig.append_trace(trace_lat, 1, 2)
# plot_url = py.plot(fig, filename='hover-chart-basic')
#py.image.save_as(fig, filename='features_PCA.png')