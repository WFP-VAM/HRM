import ee
import urllib
from io import BytesIO
import os


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
        self.date_from = date_from
        self.date_end = date_end
        self.scope = scope

    def download(self):
        print('INFO: downloading rms indexes for area of interest ...')

        if os.path.exists(self.dir + str(self.area["coordinates"]) + "NDVI_max.tif") \
        and os.path.exists(self.dir + str(self.area["coordinates"]) + "NDBI_max.tif") \
        and os.path.exists(self.dir + str(self.area["coordinates"]) + "NDWI_max.tif"):
            self.files = ["NDVI_max.tif", "NDBI_max.tif", "NDWI_max.tif"]
            print('INFO: NDs data for {} already downloaded'.format(self.area["coordinates"]))
        else:

            ee.Initialize()
            GREEN = 'B3'
            RED = 'B4'
            NIR = 'B8'
            SWIR = 'B11'
            sentinel = ee.ImageCollection('COPERNICUS/S2') \
                .filterDate(ee.Date(self.date_from), ee.Date(self.date_end)) \
                .filterBounds(self.area) \
                .select([GREEN, RED, NIR, SWIR]) \
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 70)

            def addIndices(image):
                ndvi = image.normalizedDifference([NIR, RED])
                ndbi = image.normalizedDifference([SWIR, NIR])
                ndwi = image.normalizedDifference([GREEN, NIR])
                return image.addBands(ndvi.rename('NDVI')) \
                    .addBands(ndbi.rename('NDBI')) \
                    .addBands(ndwi.rename('NDWI'))

            img = sentinel.map(addIndices).select(['NDVI', 'NDBI', 'NDWI']).reduce("max").clip(self.area)

            # download
            if self.scope == 'urban':
                print('INFO: NDs scope > urban')
                scale = 100
            else:
                print('INFO: NDs scope -> country')
                scale = 5000
            url = img.getDownloadUrl({'crs': 'EPSG:4326', 'region': self.area, 'scale': scale})
            self.files = self.unzip(BytesIO(urllib.request.urlopen(url).read()), self.dir, self.area)

    @staticmethod
    def unzip(buffer, dir, area):

        from zipfile import ZipFile

        zip_file = ZipFile(buffer)
        files = zip_file.namelist()
        for i, j in zip(files[-3:], ["NDVI_max.tif", "NDBI_max.tif", "NDWI_max.tif"]):
            print(i, j)
            zip_file.extract(i, dir)
            os.rename(dir + i, dir + str(area["coordinates"]) + j)

        return ["NDVI_max.tif", "NDBI_max.tif", "NDWI_max.tif"]

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
                build = NDBI.read(1)[i, j]
                wat = NDWI.read(1)[i, j]
                return veg, build, wat

            except IndexError as e:
                print(e)
                pass

        return df.apply(val_extract, axis=1)
