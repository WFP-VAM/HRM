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


