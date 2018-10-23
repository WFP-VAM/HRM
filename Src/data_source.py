# -*- coding: utf-8 -*-#
import os


class DataSource:

    """ Abstract class that handles a data source.

    Each data source in HRM should inherit from this class and
    overload the attributes and methods accordingly.

    Attributes:
        directory (str): where the data of the class will be saved."""

    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def download(self, lon, lat):
        """ Download the data relevant to the area of interest (aoi)

        Args:
            lon (list): list of longitudes.
            lat (lsit): list of latitudes.
        """
        pass

    def featurize(self,  lon, lat):
        """ Returns the covariate for each location
        Args:
            lon (list): list of longitudes.
            lat (lsit): list of latitudes.
        Returns:
            covariates for each lon/lat pair.
        """
        pass
