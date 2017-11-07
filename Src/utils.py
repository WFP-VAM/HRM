import pandas as pd

def scoring_postprocess(features):
    # postprocess
    features = features.transpose().reset_index()
    features["i"] = features["index"].str.slice(0, 5)
    features["j"] = features["index"].str.slice(6, 10)
    features["i"] = pd.to_numeric(features["i"])
    features["j"] = pd.to_numeric(features["j"])

    return features


# get all coordinates for country
import shapefile
from shapely.geometry import shape, Point
def get_coordinates_of_country(path_to_shape_file, spacing=1):
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
    for lat in range(-35, 60, spacing):
        for lon in range(-120, 150, spacing):
            if check(lon, lat):
                list_lon.append(lon)
                list_lat.append(lat)
    return list_lat, list_lon