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
    import ee
    ee.Initialize()
    #Functions to create new bands to add the collection
    GREEN = 'B3'
    RED = 'B4'
    NIR = 'B8'
    SWIR = 'B11'

    sentinel = ee.ImageCollection('COPERNICUS/S2') \
        .filterDate(start_date, end_date) \
        .filterBounds(geojson) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 30)

    def addIndices(image):
        ndvi = image.normalizedDifference([NIR, RED])
        ndbi = image.normalizedDifference([SWIR, NIR])
        ndwi = image.normalizedDifference([GREEN, NIR])
        return image.addBands(ndvi.rename('NDVI')).addBands(ndbi.rename('NDBI')).addBands(ndwi.rename('NDWI'))

    sentinel_w_indices = sentinel.map(addIndices)

    maxImageSentinel = sentinel_w_indices.select(['NDBI', 'NDVI', 'NDWI']).max()

    path = maxImageSentinel.getDownloadUrl({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': geojson
    })
    return path


def gee_NDBI_NDVI_NDWI(geojson, start_date="2017-01-01", end_date="2018-01-01"):
    import ee
    ee.Initialize()

    # Functions to create new bands to add the collection
    GREEN = 'B3'
    RED = 'B4'
    NIR = 'B8'
    SWIR = 'B11'

    sentinel = ee.ImageCollection('COPERNICUS/S2') \
        .filterDate(start_date, end_date) \
        .filterBounds(geojson) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 30)

    def addIndices(image):
        ndvi = image.normalizedDifference([NIR, RED])
        ndbi = image.normalizedDifference([SWIR, NIR])
        ndwi = image.normalizedDifference([GREEN, NIR])
        return image.addBands(ndvi.rename('NDVI')).addBands(ndbi.rename('NDBI')).addBands(ndwi.rename('NDWI'))

    NDBI_NDVI_NDWI = sentinel.map(addIndices).select(['NDBI', 'NDVI', 'NDWI'])

    NDBI_NDVI_NDWI_min = NDBI_NDVI_NDWI.min().reduceRegion(reducer=ee.Reducer.sum(), geometry=geojson, crs='EPSG:4326', scale=1).getInfo()
    NDBI_NDVI_NDWI_max = NDBI_NDVI_NDWI.max().reduceRegion(reducer=ee.Reducer.sum(), geometry=geojson, crs='EPSG:4326', scale=1).getInfo()
    NDBI_NDVI_NDWI_mean = NDBI_NDVI_NDWI.mean().reduceRegion(reducer=ee.Reducer.sum(), geometry=geojson, crs='EPSG:4326', scale=1).getInfo()
    NDBI_min = NDBI_NDVI_NDWI_min["NDBI"]
    NDBI_max = NDBI_NDVI_NDWI_max["NDBI"]
    NDBI_mean = NDBI_NDVI_NDWI_mean["NDBI"]
    NDVI_min = NDBI_NDVI_NDWI_min["NDVI"]
    NDVI_max = NDBI_NDVI_NDWI_max["NDVI"]
    NDVI_mean = NDBI_NDVI_NDWI_mean["NDVI"]
    NDWI_min = NDBI_NDVI_NDWI_min["NDWI"]
    NDWI_max = NDBI_NDVI_NDWI_max["NDWI"]
    NDWI_mean = NDBI_NDVI_NDWI_mean["NDWI"]

    NDVI_mean = NDBI_NDVI_NDWI_mean["NDVI"]
    return NDBI_min, NDBI_max, NDBI_mean, NDVI_min, NDVI_max, NDVI_mean, NDWI_min, NDWI_max, NDWI_mean

def download_and_unzip(buffer, a, b, path):
    unzipped = []
    from zipfile import ZipFile

    zip_file = ZipFile(buffer)
    files = zip_file.namelist()
    for i in range(a,b):
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
