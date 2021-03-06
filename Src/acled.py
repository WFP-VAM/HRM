# -*- coding: utf-8 -*-#
from data_source import DataSource
import os
import pandas as pd
import requests
import io
import geopandas as gpd
from shapely.geometry import Point, Polygon
from osmnx.core import bbox_from_point
import numpy as np


class ACLED(DataSource):

    def __init__(self, directory):
        DataSource.__init__(self, directory)

        """ Overload the directory path. """
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.path = None

    def download(self, country_ISO, date_from, date_to):
        """ Downloads all the ACLED events for a given country and date range
        Group the events by unique pair of coordinates and calculate:
            - The number of unique events
            - The sum of fatalities
            - The number of events of type Violence against civilians
        Save the resulting geojson
        Args:
            country_ISO: ISO number of the country of interest. Get them from https://www.acleddata.com/wp-content/uploads/dlm_uploads/2017/10/API-User-Guide-11.pdf
            date_from (str): consider only events from this date.
            date_to (str): consider only events up to this date.
        """
        self.path = os.path.join(
            self.directory,
            "events_{}_{}_{}.json".format(country_ISO, date_from, date_to))

        if os.path.exists(self.path):
            print('INFO: ACLED data already downloaded')
            return

        URL = "https://api.acleddata.com/acled/read.csv"
        parameters = {
            "terms": "accept",
            "iso": country_ISO,
            "fields": "iso|event_type|fatalities|latitude|longitude",
            "event_date": "{" + date_from + "|" + date_to + "}",
            "event_date_where": "BETWEEN"
        }

        data = requests.get(URL, params=parameters).content

        data = pd.read_csv(io.StringIO(data.decode('utf-8')))

        print("INFO: downloaded ", len(data), "ACLED events")
        if len(data) == 0:
            return

        sum_fatalities = data.groupby(['latitude', 'longitude'])["fatalities"].sum()
        count_all_events = data.groupby(['latitude', 'longitude']).size().rename("n_events")
        count_violence_civ = data[data['event_type'] == 'Violence against civilians'].groupby(['latitude', 'longitude']).size().rename("violence_civ")

        results = pd.concat([sum_fatalities, count_all_events, count_violence_civ], axis=1)

        results["violence_civ"] = results["violence_civ"].fillna(0).astype(int)

        results = results.reset_index()

        results['Coordinates'] = list(zip(results['longitude'], results['latitude']))
        results['Coordinates'] = results['Coordinates'].apply(Point)

        gdf = gpd.GeoDataFrame(results, geometry='Coordinates')

        gdf.crs = {'init': 'epsg:4326'}
        gdf.to_file(self.path, driver='GeoJSON')
        print("INFO: Saved ", len(results), "ACLED unique GPS pairs")

    @staticmethod
    def __sum_within_bbox(lat, lon, buffer, gdf, property):
        """
        Gets the sum of a geojson MultiPoint property within the bbox around a pair of coordinates.
        """
        point = (lat, lon)
        north, south, east, west = bbox_from_point(point, buffer)
        point1 = (east, south)
        point2 = (west, south)
        point3 = (west, north)
        point4 = (east, north)
        poly = Polygon([point1, point2, point3, point4])
        density = gdf[gdf.within(poly)][property].sum()
        return density

    def featurize(self, longitudes, latitudes, function, property=None, buffer=50000):
        """
        Computes features from a geodataframe for a list of longitudes and latitudes.
        For each lat, lon it:
            If function equal to density: computes the sum of a specific attribute for the points
             within a bbox of side length equal to buffer around the .
            If function equal to distance: computes the distance to the closest point.
            If function equal to weighted_kNN: computes the sum of the 10 closest point for a specific attribute
            weighted by the distance to the points.
        """
        if os.path.exists(self.path):
            gdf = gpd.read_file(self.path)
        else:
            print('ACLED not downloaded or zero events.')
            return
        features = []
        if function == 'density':
            for lat, lon in zip(latitudes, longitudes):
                density = self.__sum_within_bbox(lat, lon, buffer, gdf, property)
                features.append(density)
        elif function == 'distance':
            from sklearn.neighbors import NearestNeighbors
            gdf_lats = gdf["geometry"].y
            gfd_lons = gdf["geometry"].x
            X = np.array([gdf_lats, gfd_lons]).T
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(X)
            features = []
            for lat, lon in zip(latitudes, longitudes):
                X = np.array([[lat, lon]])
                a, _ = nbrs.kneighbors(X)
                features.append(a[[0]][0][0])
        elif function == 'weighted_kNN':
            from sklearn.neighbors import NearestNeighbors
            gdf_lats = gdf["geometry"].y
            gfd_lons = gdf["geometry"].x
            X = np.array([gdf_lats, gfd_lons]).T
            y = np.array(gdf[property])
            nbrs = NearestNeighbors(n_neighbors=10).fit(X)
            for lat, lon in zip(latitudes, longitudes):
                X_pred = np.array([[lat, lon]])
                dist, c = nbrs.kneighbors(X_pred)  # distance (in degrees) and index of the nearest neighbor
                dist = dist / 10
                feature = (y[c][dist != 0] * 1 / (1 + dist[dist != 0])).sum()  # weight by the inverse of the distance
                features.append(feature)

        return features
