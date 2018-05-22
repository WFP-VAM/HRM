import pandas as pd
import gdal
import shapefile
from shapely.geometry import shape, Point
import numpy as np


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


def tifgenerator(outfile, raster_path, df):
    """
    Given a filepath (.tif), a raster for reference and a dataset with i, j and yhat
    it generates a raster.
    :param outfile:
    :param raster_path:
    :param df:
    :return:
    """

    print('-> writing: ', outfile)
    # create empty raster from the original one
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape
    arr_out = np.zeros(arr.shape) - 99
    arr_out[df['j'], df['i']] = df['yhat']
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)

    outdata.SetGeoTransform(ds.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input

    outdata.GetRasterBand(1).SetNoDataValue(-99)
    outdata.GetRasterBand(1).WriteArray(arr_out)

    outdata.FlushCache()  # saves to disk!!


def aggregate(input_rst, output_rst, scale):
    """
    Downsample (upscale) a raster by a given factor and replace no_data value with 0.
    Args:
        input_rst: path to the input raster in a format supported by georaster
        output_rst: path to the scaled output raster in a format supported by georaster
        scale: The scale (integer) by which the raster in upsampeld.
    Returns:
        Save the output raster to disk.
    # https://github.com/pasquierjb/GIS_RS_utils/blob/master/aggregate_results.py
    """
    import georasters as gr
    input_gr = gr.from_file(input_rst)

    # No data values are replaced with 0 to prevent summing them in each block.
    input_gr.raster.data[input_gr.raster.data == input_gr.nodata_value] = 0
    input_gr.nodata_value = 0

    output_gr = input_gr.aggregate(block_size=(scale, scale))

    output_gr.to_tiff(output_rst.replace(".tif", ""))


def squaretogeojson(lon, lat, d):
    from math import pi,cos
    from geojson import Polygon
    r_earth=6378000
    minx  = lon  - ((d/2) / r_earth) * (180 / pi)
    miny = lat - ((d/2) / r_earth) * (180 / pi) / cos(lon * pi/180)
    maxx  = lon  + ((d/2) / r_earth) * (180 / pi)
    maxy = lat + ((d/2) / r_earth) * (180 / pi) / cos(lon * pi/180)
    #return minx,miny,maxx,maxy
    square=Polygon([[(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]])
    return square