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


# ----------------------------------------------------------------------
# import and process the features created

features = pd.read_csv('../Data/Intermediate_files/google_sat_CNN_features_lsms_ResNet_tf_last.csv')

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
GRId = RasterGrid()
dfx['i'], dfx['j'] = features['i'], features['j']

# for what clusters?
GRId.config["dataset"]["filename"]
dfx['lon'] , dfx['lat']  = GRId.get_gpscoordinates(dfx['i'], dfx['j'])
dfx['lonlat'] = dfx[['lon', 'lat']].round(2).astype(str).apply(lambda x: ','.join(x), axis = 1)


# ----------------------------------------------------------------------
# export for Viz
dfx.to_clipboard()


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