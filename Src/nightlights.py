# -*- coding: utf-8 -*-#
from data_source import DataSource
import os
import ee
import urllib
from io import BytesIO


class Nightlights(DataSource):
    """overloading the DataSource class."""

    def __init__(self, directory):
        DataSource.__init__(self, directory)

        self.build_threshold = 0.3
        self.file = None

        """ Overload the directory path. """
        self.directory = os.path.join(self.directory, 'nightlights/')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def download(self, area, date_from, date_end):
        """ given an area defined by a geoJSON, it downloads a nightlights raster at the specified date at the most granular level (500x500)
        Args:
            area: geoJSON, use src.points_to_poligon to generate.
            date_from (str): consider only lights from this date.
            date_end (str): consider only lights up to this date.
        """

        print('INFO: downloading nightlights for area of interest ...')
        ee.Initialize()
        start = ee.Date(date_from)
        end = ee.Date(date_end)

        # Worldpop's population mask
        popMask = ee.ImageCollection("WorldPop/POP").select('population').reduce('median').gt(0.3)
        # VIIRS monthly product
        NLImgSet = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').select('avg_rad').filterDate(start, end)

        # Mask each NL image by WorldPop's layer
        def maskImage(img):
            return img.updateMask(popMask)

        NLImg = NLImgSet.map(maskImage).sum()

        # download
        url = NLImg.getDownloadURL({'crs': 'EPSG:4326', 'region': area, 'scale': 1000})
        self.file = self.directory+self.download_and_unzip(BytesIO(urllib.request.urlopen(url).read()), self.directory)

    def featurize(self, lon, lat):
        """ Given lon lat lists, it returns the nightlight value at each point.
        Args:
            lon (list): list of longitudes.
            lat (list): list of latitudes.
        """

        import rasterio
        try:
            pop = rasterio.open(self.file)
        except MemoryError:
            print('Landuse Raster too big!')
            raise

        nightlights = []
        for lon_val, lat_val in zip(lon, lat):

            try:
                i, j = pop.index(lon_val, lat_val)
                nightlights.append(pop.read(1)[i, j])

            except IndexError as e:
                print(e, lon_val, ", ", lat_val)
                nightlights.append(0)

        return nightlights

    @staticmethod
    def download_and_unzip(buffer, dir):

        from zipfile import ZipFile

        zip_file = ZipFile(buffer)
        files = zip_file.namelist()
        for i in files:
            zip_file.extract(i, dir)

        return files[1]


