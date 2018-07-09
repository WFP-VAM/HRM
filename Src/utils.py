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
            if pol.contains(Point(lo, lat)):
                v.append(val)
        return np.mean(v)

    # loop over the polygons
    shp = shapefile.Reader(path_to_shape_file)
    value_means = []
    for poly in shp.shapes():
        polygon = shape(poly)
        value_means.append(vals_in_poly(lon, lat, val, polygon))
    return value_means


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
    print(len(input_gr.raster.data.astype(np.float32) == np.float32(input_gr.nodata_value)))
    input_gr.raster.data[input_gr.raster.data.astype(np.float32) == np.float32(input_gr.nodata_value)] = 0
    input_gr.nodata_value = 0

    output_gr = input_gr.aggregate(block_size=(scale, scale))

    output_gr.to_tiff(output_rst.replace(".tif", ""))

def upscaleBaseRaster(base_raster, aggregate_factor, dataset_df, minimum_pop=0.3, minlat=None, maxlat=None, minlon=None, maxlon=None):
    """
    reduce resolution of base raster
    Args:
        base_raster: filepath to base raster
        aggregat_factor: self explantory
        dataset_df: dataframe of dataset for scoring with GPS coords
        minimum_pop: population density below which areas are omitted
    """
    # WorldPop Raster too fine, aggregate #
    #if aggregate_factor is None:
    #    aggregate_factor = config["base_raster_aggregation"][0]

    if aggregate_factor > 1:
        print('INFO: aggregating raster ...')
        base_raster = "../tmp/local_raster.tif"
        aggregate(raster, base_raster, aggregate_factor)
    else:
        base_raster = raster

    # ---------------- #
    # AREA OF INTEREST #
    # ---------------- #
    data_cols = dataset_df.columns.values

    # create geometry
    if (minlat is None) and (maxlat is None) and (minlon is None) and (maxlon is None):
        minlat, maxlat, minlon, maxlon = df_boundaries(dataset_df, buffer=0.05, lat_col="gpsLatitude", lon_col="gpsLongitude")

    area = points_to_polygon(minlon, minlat, maxlon, maxlat)

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

    final_raster = "../tmp/final_raster.tif"
    with rasterio.open(final_raster, "w", **out_meta) as dest:
        out_image[out_image < minimum_pop] = dest.nodata
        dest.write(out_image)
        list_j, list_i = np.where(dest.read()[0] != dest.nodata)

    return final_raster
    
    
    
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
    start = datetime.strptime(start,"%Y-%m-%d")
    end = datetime.strptime(end,"%Y-%m-%d")
    diff = (end  - start ) / intv
    for i in range(intv):
        yield ((start + diff * i).strftime("%Y-%m-%d"),(start + diff * (i+1)).strftime("%Y-%m-%d"))
