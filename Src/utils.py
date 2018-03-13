import pandas as pd

def scoring_postprocess(features):
    # postprocess
    features = features.transpose().reset_index()
    features["i"] =  features["index"].apply(lambda x: x.split('_')[0])
    features["j"] =  features["index"].apply(lambda x: x.split('_')[1])
    features["i"] = pd.to_numeric(features["i"])
    features["j"] = pd.to_numeric(features["j"])

    return features


# get all coordinates for country
import shapefile
from shapely.geometry import shape, Point
import numpy as np

def get_coordinates_from_shp(path_to_shape_file, spacing=1):
    """
    function that given a shapefile it return all the coordinates within the boundary. You can chose the coordinates
    steps over which it  checks (ex. integers -> spacings=1)
    :param path_to_shape_file: string
    :param spacing: float or int. Default to 1
    :return: two lists of latitudes and longitudes
    """

    shapes = shapefile.Reader(path_to_shape_file).shapes()
    polygon = shape(shapes[0])

    def check(lon, lat):
        # build a shapely point from your geopoint
        point = Point(lon, lat)
        # the contains function checks that
        return polygon.contains(point)

    list_lat = []
    list_lon = []
    for lat in np.arange(-35, 60, spacing):
        for lon in np.arange(-120, 150, spacing):
            if check(lon, lat):
                list_lon.append(lon)
                list_lat.append(lat)
    return list_lat, list_lon

def zonal_stats(path_to_shape_file, lon, lat, val):
    """
    given a shapefile and a list of coordinates and values, it returns the mean of the values for
    each polygoon.
    :param path_to_shape_file: string, path to shapefile
    :param lon: list of longitudes
    :param lat: list of latitudes
    :param val: list of values
    :return: list of means (one for each polygon)
    """

    def vals_in_poly(lon, lat, vals, pol):
        v = []
        for lo, la, val in zip(lon, lat, vals):
            if pol.contains(Point(lo,la)):
                v.append(val)
        return np.mean(v)

    # loop over the polygons
    shp = shapefile.Reader(path_to_shape_file)
    value_means = []
    for poly in shp.shapes():
        polygon = shape(poly)
        value_means.append(vals_in_poly(lon, lat, val, polygon))
    return value_means


def shape2json(fname, outfile="states.json"):
    """
    function to convert a shapefile to GeoJSON.
    Similar to: http://geospatialpython.com/2013/07/shapefile-to-geojson.html
    :param fname: shp file path
    :param outfile: json file path
    """
    # read the shapefile
    reader = shapefile.Reader(fname)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]
    buffer = []
    for sr in reader.shapeRecords():
        atr = dict(zip(field_names, sr.record))
        geom = sr.shape.__geo_interface__
        try:
            buffer.append(dict(id=str(atr['ID_0'])+'_'+str(atr['ID_1'])+'_'+str(atr['ID_2']), type="Feature", \
                           geometry=geom, properties=atr))
        except KeyError:
            buffer.append(dict(id=str(atr['ID_0']) + '_' + str(atr['ID_1']), type="Feature", \
                               geometry=geom, properties=atr))

    # write the GeoJSON file
    from json import dumps
    geojson = open(outfile, "w")
    geojson.write(dumps({"type": "FeatureCollection", \
                         "features": buffer}, indent=2) + "\n")
    geojson.close()


