import pandas as pd
import gdal
from shapely.geometry import shape, Point
import numpy as np


def tifgenerator(outfile, raster_path, df, value='yhat'):
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
    arr_out[df['j'], df['i']] = df[value]
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
    input_gr.raster.data[input_gr.raster.data.astype(np.float32) == np.float32(input_gr.nodata_value)] = 0
    input_gr.nodata_value = 0

    output_gr = input_gr.aggregate(block_size=(scale, scale))

    output_gr.to_tiff(output_rst.replace(".tif", ""))


def squaretogeojson(lon, lat, d):
    from math import pi, cos
    r_earth = 6378000
    minlon = lon - ((d / 2) / r_earth) * (180 / pi)
    minlat = lat - ((d / 2) / r_earth) * (180 / pi) / cos(lon * pi / 180)
    maxlon = lon + ((d / 2) / r_earth) * (180 / pi)
    maxlat = lat + ((d / 2) / r_earth) * (180 / pi) / cos(lon * pi / 180)
    #return minx,miny,maxx,maxy
    square = points_to_polygon(minlon, minlat, maxlon, maxlat)
    return square


def df_boundaries(df, buffer=0.05, lat_col="gpsLatitude", lon_col="gpsLongitude"):
    '''
    Get GPS coordinates of the boundary box of a DataFrame and add some buffer around it.
    '''
    from numpy import round
    minlat = df["gpsLatitude"].min()
    maxlat = df["gpsLatitude"].max()
    minlon = df["gpsLongitude"].min()
    maxlon = df["gpsLongitude"].max()

    lat_buffer = (maxlat - minlat) * buffer
    lon_buffer = (maxlon - minlon) * buffer

    minlat = round(minlat - lat_buffer, 5)
    maxlat = round(maxlat + lat_buffer, 5)
    minlon = round(minlon - lon_buffer, 5)
    maxlon = round(maxlon + lon_buffer, 5)

    return minlat, maxlat, minlon, maxlon


def points_to_polygon(minlon, minlat, maxlon, maxlat):
    from geojson import Polygon
    square = Polygon([[(minlon, minlat), (maxlon, minlat), (maxlon, maxlat), (minlon, maxlat), (minlon, minlat)]])
    return square


def multiply(input_rst1, input_rst2, output_rst):
    """
    Multiply two rasters cell by cell.
    Args:
        input_rst1: path to the input raster in a format supported by gdal
        input_rst2: path to the output raster in a format supported by gdal
        output_rst: path to the raster to copy the referenced system (projection and transformation) from
    Returns:
        Save the output raster to disk.
    """
    import rasterio

    with rasterio.open(input_rst1) as src1:
        with rasterio.open(input_rst2) as src2:
            data1 = src1.read()
            data2 = src2.read()
            data1[data1 == src1.nodata] = 0
            data2[data2 == src2.nodata] = 0
            final = data1 * data2
            profile = src1.profile

    with rasterio.open(output_rst, 'w', **profile) as dst:
        dst.nodata = 0
        dst.write(final)


def weighted_sum_by_polygon(input_shp, input_rst, weight_rst, output_shp):
    """
    Take the weighted sum of two rasters (indicator and weights) within each polygon of a shapefile.

    input_rst and weight_rst need to be in the same projection system and have the same shape

    Args:
        input_shp: path to the input shapefile in a format support by geopandas
        input_rst: path to the raster containing the value you want to aggregate
        weight_rst: path to the raster containing the weights (ex: population data)
        output_shp: path to the input shapefile in a format support by geopandas
    Returns:
        Save a copy of the shapefile to disk with the resulting weighted sum as a new attribute of each polygon.
    """
    import geopandas as gpd
    import rasterio
    import rasterio.mask
    import json
    import numpy as np

    mult_rst = input_rst.replace(".tif", "_multiplied.tif")
    multiply(input_rst, weight_rst, mult_rst)

    X = []
    Y = []
    gdf = gpd.read_file(input_shp)
    #gdf['indicator'] = None
    #gdf['population'] = None

    with rasterio.open(mult_rst) as src1:
        with rasterio.open(weight_rst) as src2:
            index = 0
            gdf = gdf.to_crs(crs=src1.crs.data)  # Re-project shape-file according to mult_rst
            features = json.loads(gdf.to_json())['features']
            for feature in features:
                geom = feature['geometry']
                try:
                    out_image, out_transform = rasterio.mask.mask(src1, [geom], crop=True)  # all_touched=False so only considers the center of each pixel
                    out_image2, out_transform2 = rasterio.mask.mask(src2, [geom], crop=True)   # all_touched=False so only considers the center of each pixel
                    out_image[out_image == src1.nodata] = 0
                    out_image2[out_image2 == src2.nodata] = 0
                    weighted_sum = out_image.sum()
                    y = out_image2.sum()
                    if y == 0:
                        x = 0.00
                    else:
                        x = (weighted_sum / y).item()
                except ValueError:
                    x = 0.00
                    y = 0

                X.append(x)
                Y.append(y)
                #print(gdf.loc[index, 'admin1Name'], x, y)
                gdf.loc[index, 'indicator'] = x
                gdf.loc[index, 'population'] = int(y)
                index += 1
    print("Total_Weights (population) : {}".format(np.array(Y).sum()))
    print('-> writing: ', output_shp)
    gdf.to_file(output_shp)


def date_range(start, end, intv):
    from datetime import datetime
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    diff = (end - start) / intv
    for i in range(intv):
        yield ((start + diff * i).strftime("%Y-%m-%d"), (start + diff * (i + 1)).strftime("%Y-%m-%d"))


def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    import time
    from functools import wraps
    """Retry calling the decorated function using an exponential backoff.

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry
