import numpy as np
import gdal
from skimage.measure import block_reduce

# create a landcover raster for a country from ESA landcover
# gdalwarp -srcnodata 0 -dstnodata -99 -crop_to_cutline -cutline C:\Users\lorenzo.riches\Downloads\SEN_adm_shp\SEN_adm0.shp esa_landcover.tif esa_landcover_Senegal_full.tif

input_raster = '../HRM/Data/Geofiles/esa_landcover_Senegal_full.tif'
outfile = '../HRM/Data/Geofiles/esa_landcover_Senegal_b_10.tif'

# load raster ---------
raster = gdal.Open(input_raster)
(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = raster.GetGeoTransform()

band = raster.GetRasterBand(1)
arr = band.ReadAsArray()

# convert landuse into buildings and no buildings -----
def buildings_or_not(a):
    if a == 8:  # building
        return 1
    elif a == -99:  # no data value
        return -99
    else:
        return 0


myfunc_vec = np.vectorize(buildings_or_not)
arr = myfunc_vec(arr)

# downsample ---------------------------------------
arr2 = block_reduce(arr, (10,10), func=np.max)

# writeout ------------------------------
print('-> writing: ', outfile)
# create empty raster from the original one
ds = gdal.Open(input_raster)
band = ds.GetRasterBand(1)
[cols, rows] = arr2.shape
print('new raster has shape: ', cols, rows)
print('with values: ', np.unique(arr2))
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Int16)

outdata.SetGeoTransform((upper_left_x, x_size*10, x_rotation, upper_left_y, y_rotation, y_size*10))  # sets same geotransform as input
outdata.SetProjection(ds.GetProjection())  # sets same projection as input

outdata.GetRasterBand(1).SetNoDataValue(-99)
outdata.GetRasterBand(1).WriteArray(arr2)

outdata.FlushCache()  # saves to disk!!
