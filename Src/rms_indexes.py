import ee
from zipfile import ZipFile
from io import BytesIO
import os
import requests


class S2indexes:

    def __init__(self, area, dir, date_from, date_end, scope):
        """
        given an area defined by a geoJSON, it returns rasters of
         remote sensing indexes at the specified date at granularuity defined by the scope parameter
        Args:
            area: geoJSON, use squaretogeojson to generate
            dir: directory where to save the easter
            date_from, date_end: nightlights for what point in time?
            scope (str): country or urban?
        """
        self.area = area
        self.dir = dir
        self.date_from = date_from
        self.date_end = date_end
        self.scope = scope
        self.files = None

    def download(self):
        print('INFO: downloading rms indexes for area of interest ...')

        if os.path.exists(self.dir + str(self.area["coordinates"]) + "NDVI_max.tif") \
        and os.path.exists(self.dir + str(self.area["coordinates"]) + "NDBI_max.tif") \
        and os.path.exists(self.dir + str(self.area["coordinates"]) + "NDWI_max.tif"):
            self.files = [str(self.area["coordinates"]) + "NDVI_max.tif", str(self.area["coordinates"]) + "NDBI_max.tif", str(self.area["coordinates"]) + "NDWI_max.tif"]
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

            for b in ['NDVI_max', 'NDBI_max', 'NDWI_max']:
                url = img.select(b).getDownloadUrl({'crs': 'EPSG:4326', 'region': self.area, 'scale': scale})
                print('url: ', url)
                r = requests.get(url)

                z = ZipFile(BytesIO(r.content))
                z.extract(z.namelist()[1], self.dir)
                os.rename(self.dir + z.namelist()[1], self.dir + str(self.area["coordinates"]) + b+'.tif')

            self.files = [str(self.area["coordinates"]) + "NDVI_max.tif", str(self.area["coordinates"]) + "NDBI_max.tif",
             str(self.area["coordinates"]) + "NDWI_max.tif"]

    def rms_values(self, longitudes, latitudes):
        """
        Given a dataset with latitude and longitude columns, it returns the nightlight value at each point.
        Args:
            longitudes: list of longitudes
            latitudes: list of latitudes
        Returns:
            Series
        """
        import rasterio
        try:
            NDVI = rasterio.open(self.dir + self.files[0])
            NDBI = rasterio.open(self.dir + self.files[1])
            NDWI = rasterio.open(self.dir + self.files[2])
        except MemoryError:
            print('Remote Sensing Indexes Raster too big!')
            raise

        veg, build, wat = [], [], []
        for lon, lat in zip(longitudes, latitudes):

            i, j = NDVI.index(lon, lat)
            veg.append(NDVI.read(1)[i, j])
            build.append(NDBI.read(1)[i, j])
            wat.append(NDWI.read(1)[i, j])

        return veg, build, wat