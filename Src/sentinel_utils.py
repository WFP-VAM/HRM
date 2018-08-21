def gee_url(geojson, start_date, end_date):
    import ee
    ee.Initialize()

    lock = 0
    cloud_cover = 10
    while lock == 0:
        sentinel = ee.ImageCollection('COPERNICUS/S2') \
            .filterDate(start_date, end_date) \
            .select('B2', 'B3', 'B4') \
            .filterBounds(geojson) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_cover) \
            .sort('GENERATION_TIME') \
            .sort('CLOUDY_PIXEL_PERCENTAGE', False)

        collectionList = sentinel.toList(sentinel.size())
        # check if there are images, otherwise increase accepteable cloud cover
        try:  # if it has zero images this line will return an EEException
            collectionList.size().getInfo()
            lock = 1
        except:  # ee.ee_exception.EEException:
            print('INFO: found no images with {}% cloud cover. Going to {}%'.format(cloud_cover, cloud_cover+10))
            cloud_cover = cloud_cover + 10

    image1 = sentinel.mosaic()
    path = image1.getDownloadUrl({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': geojson
    })
    return path


def gee_maxNDBImaxNDVImaxNDWI_url(geojson, start_date, end_date):

    maxImageSentinel = gee_sentinel_raster(start_date, end_date, geojson, agg="max", ind="NDVI")

    path = maxImageSentinel.getDownloadUrl({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': geojson
    })
    return path


def gee_sentinel_raster(start_date, end_date, large_area, agg="max", ind="NDVI"):
    import ee
    ee.Initialize()
    # Functions to create new bands to add the collection
    GREEN = 'B3'
    RED = 'B4'
    NIR = 'B8'
    SWIR = 'B11'

    sentinel = ee.ImageCollection('COPERNICUS/S2') \
        .filterDate(start_date, end_date) \
        .filterBounds(large_area) \
        .select(['B3', 'B4', 'B8', 'B11'])

    def addIndices(image):
        ndvi = image.normalizedDifference([NIR, RED])
        ndbi = image.normalizedDifference([SWIR, NIR])
        ndwi = image.normalizedDifference([GREEN, NIR])
        return image.addBands(ndvi.rename('NDVI')).addBands(ndbi.rename('NDBI')).addBands(ndwi.rename('NDWI'))

    sentinel_w_indices = sentinel.map(addIndices)

    maxraster = sentinel_w_indices.select(ind).reduce(agg).clip(large_area)

    return maxraster


def gee_raster_mean(df, gee_raster, lat_col="gpsLatitude", lon_col="gpsLongitude", ind="NDVI", agg="max"):
    from utils import squaretogeojson
    import ee
    small_area = squaretogeojson(df[lon_col], df[lat_col], 100)
    value = gee_raster.reduceRegion(reducer=ee.Reducer.mean(), geometry=small_area, crs='EPSG:4326', scale=10).getInfo()
    if len(value) == 0:
        print("INFO: GEE, No data at this location")
    else:
        return value[ind + "_" + agg]


def download_and_unzip(buffer, a, b, path):
    unzipped = []
    import zipfile
    try:
        zip_file = zipfile.ZipFile(buffer)
    except zipfile.BadZipFile:  # Often happens with GEE API
        print("bad_zip")
        return None
    files = zip_file.namelist()
    for i in range(a, b):
        zip_file.extract(files[i], path + "/tiff/")
        #print("{} downloaded and unzippped".format(files[i]))
        unzipped.append(files[i])
    return unzipped


def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min) / (band_max - band_min))


def rgbtiffstojpg(files, path, name):
    '''
    files: a list of files ordered as follow. 0: Blue Band 1: Green Band 2: Red Band
    path: the path to look for the tiff files and to save the jpg
    '''
    import scipy.misc as sm
    import gdal
    import numpy as np
    b2_link = gdal.Open(path + "/tiff/" + files[0])
    b3_link = gdal.Open(path + "/tiff/" + files[1])
    b4_link = gdal.Open(path + "/tiff/" + files[2])

    # call the norm function on each band as array converted to float
    b2 = norm(b2_link.ReadAsArray().astype(np.float))
    b3 = norm(b3_link.ReadAsArray().astype(np.float))
    b4 = norm(b4_link.ReadAsArray().astype(np.float))

    # Create RGB
    rgb = np.dstack((b4, b3, b2))
    del b2, b3, b4
    sm.toimage(rgb, cmin=np.percentile(rgb, 2), cmax=np.percentile(rgb, 98)).save(path + name)
