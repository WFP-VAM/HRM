import ee
import urllib
from io import BytesIO
from pandas import to_datetime


class Nightlights:

    def __init__(self, area, dir, date_from, date_end):
        """
        given an area defined by a geoJSON, it returns a nightlights raster at the specified date at the most granular level (500x500)
        :param area: geoJSON, use squaretogeojson to generate
        :param dir: directory where to save the easter
        :param date: nightlights for what point in time?
        """
        self.area = area
        self.dir = dir
        self.build_threshold = 0.3

        print('INFO: downloading nightlights for area of interest ...')
        ee.Initialize()
        start = ee.Date(date_from)
        end = ee.Date(date_end)

        # Create mask using DMSP-OLS to select settlements -----------------
        popImgSet = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS').\
            select('stable_lights').\
            filterDate('2010-01-01', date_end)

        popImgmask = ee.Image(popImgSet.sort('system:index', False).first())

        # Select VIIRS Nightlights images ----------------------------------
        NLImgSet = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').select('avg_rad').filterDate(start, end)

        # Mask each NL image first by DMSP-OLS then to only select values greater than 0.3
        def maskImage(img):
            return img.updateMask(popImgmask).updateMask(img.gte(0.3))

        NLImgSet = NLImgSet.map(maskImage)

        # from collection to 1 image
        first = ee.Image(NLImgSet.first())

        def appendBand(img, previous):
                return ee.Image(previous).addBands(img)

        img = ee.Image(NLImgSet.iterate(appendBand, first))

        # reduce (mean) pixel wise
        img = img.reduce(ee.Reducer.mean())

        # download
        url = img.getDownloadURL({'crs': 'EPSG:4326', 'region': area})
        self.file = self.dir+self.download_and_unzip(BytesIO(urllib.request.urlopen(url).read()), dir)

    @staticmethod
    def download_and_unzip(buffer, dir):

        from zipfile import ZipFile

        zip_file = ZipFile(buffer)
        files = zip_file.namelist()
        for i in files:
            zip_file.extract(i, dir)

        return files[1]

    def nightlights_values(self, df, lon_col='gpsLongitude', lat_col='gpsLatitude'):
        """
        Given a dataset with latitude and longitude columns, it returns the nightlight value at each point.
        :param df: DataFrame
        :param lon_col: column names for longitude
        :param lat_col: column name of latitude
        :return: Series
        """

        import rasterio
        try:
            pop = rasterio.open(self.file)
        except MemoryError:
            print('Landuse Raster too big!')
            raise

        def lu_extract(row):

            try:
                i, j = pop.index(row[lon_col], row[lat_col])
                lu = pop.read(1)[i, j]
                return lu

            except IndexError as e:
                print(e, row[lon_col], ", ", row[lat_col])
                lu = 0
                return lu

        return df.apply(lu_extract, axis=1)
