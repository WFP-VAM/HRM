import ee
import urllib
from io import BytesIO
from pandas import to_datetime


class Nightlights:

    def __init__(self, area, dir, date):
        """
        given an area defined by a geoJSON, it returns a nightlights raster at the specified date at the most granular level (500x500)
        :param area: geoJSON, use squaretogeojson to generate
        :param dir: directory where to save the easter
        :param date: nightlights for what point in time?
        """
        self.area = area
        self.dir = dir
        self.date = date
        self.build_threshold = 0.3

        print('INFO: downloading nightlights for area of interest ...')
        ee.Initialize()
        now = ee.Date(date)
        # Create mask using DMSP-OLS to select settlements -----------------
        popImgSet = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS').select('stable_lights').filterDate('2010-01-01',
                                                                                                            now)
        popImgmask = ee.Image(popImgSet.sort('system:index', False).first())
        #popImgmask = popImg.gte(self.build_threshold)

        # Select VIIRS Nightlights images ----------------------------------
        NLImgSet = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').select('avg_rad').filterDate('2014-01-01', now)

        # Mask each NL image first by DMSP-OLS then to only select values greater than 0.3
        def maskImage(img):
            return img.updateMask(popImgmask).updateMask(img.gte(0.3))

        NLImgSet = NLImgSet.map(maskImage)

        url = ee.Image(NLImgSet.sort('system:index', False).first()).getDownloadURL({'crs': 'EPSG:4326', 'region': area})
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

        import georasters as gr
        try:
            pop = gr.load_tiff(self.file)
        except MemoryError:
            print('Landuse Raster too big!')
            raise

        # Find location of point (x,y) on raster, e.g. to extract info at that location
        NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(self.file)

        def lu_extract(row):

            try:
                c, r = gr.map_pixel(row[lon_col], row[lat_col], GeoT[1], GeoT[-1], GeoT[0], GeoT[3])
                lu = pop[c, r]
                return lu

            except IndexError:
                pass

        return df.apply(lu_extract, axis=1)
