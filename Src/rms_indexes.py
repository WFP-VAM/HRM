import ee
import urllib
from io import BytesIO


class S2indexes:

    def __init__(self, area, dir, date_from, date_end, scope):
        """
        given an area defined by a geoJSON, it returns rasters of
         remote sensing indexes at the specified date at granularuity defined by the scope parameter
        :param area: geoJSON, use squaretogeojson to generate
        :param dir: directory where to save the easter
        :param date: nightlights for what point in time?
        """
        self.area = area
        self.dir = dir

        print('INFO: downloading rms indexes for area of interest ...')
        ee.Initialize()
        GREEN = 'B3'
        RED = 'B4'
        NIR = 'B8'
        SWIR = 'B11'
        sentinel = ee.ImageCollection('COPERNICUS/S2') \
            .filterDate(ee.Date(date_from), ee.Date(date_end)) \
            .filterBounds(area) \
            .select([GREEN, RED, NIR, SWIR]) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 70)

        def addIndices(image):
            ndvi = image.normalizedDifference([NIR, RED])
            ndbi = image.normalizedDifference([SWIR, NIR])
            ndwi = image.normalizedDifference([GREEN, NIR])
            return image.addBands(ndvi.rename('NDVI')) \
                .addBands(ndbi.rename('NDBI')) \
                .addBands(ndwi.rename('NDWI'))

        img = sentinel.map(addIndices).reduce("max").clip(area)

        # download
        if scope == 'urban':
            print('INFO: NDs scope > urban')
            scale = 100
        else:
            print('INFO: NDs scope -> country')
            scale = 5000
        url = img.getDownloadUrl({'crs': 'EPSG:4326', 'region': area, 'scale': scale})
        self.files = self.download_and_unzip(BytesIO(urllib.request.urlopen(url).read()), dir)

    @staticmethod
    def download_and_unzip(buffer, dir):

        from zipfile import ZipFile

        zip_file = ZipFile(buffer)
        files = zip_file.namelist()
        for i in files[-3:]:
            zip_file.extract(i, dir)

        return files[-3:]

    def rms_values(self, df, lon_col='gpsLongitude', lat_col='gpsLatitude'):
        """
        Given a dataset with latitude and longitude columns, it returns the nightlight value at each point.
        :param df: DataFrame
        :param lon_col: column names for longitude
        :param lat_col: column name of latitude
        :return: Series
        """
        import rasterio
        try:
            NDVI = rasterio.open(self.dir + self.files[0])
            NDBI = rasterio.open(self.dir + self.files[1])
            NDWI = rasterio.open(self.dir + self.files[2])
        except MemoryError:
            print('Remote Sensing Indexes Raster too big!')
            raise

        def val_extract(row):

            try:  # TODO: BUFFER!
                i, j = NDVI.index(row[lon_col], row[lat_col])
                veg = NDVI.read(1)[i, j]
                burn = NDBI.read(1)[i, j]
                wat = NDWI.read(1)[i, j]

                return veg, burn, wat

            except IndexError:
                pass

        return df.apply(val_extract, axis=1)
