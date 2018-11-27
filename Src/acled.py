# -*- coding: utf-8 -*-#
from data_source import DataSource
import os
import pandas as pd
import requests
import io
import geopandas as gpd
from shapely.geometry import Point, Polygon
from osmnx.core import bbox_from_point


class ACLED(DataSource):

    def __init__(self, dir, country_ISO, date_from, date_to):
        self.country_ISO = country_ISO
        self.date_from = date_from
        self.date_to = date_to
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.path = os.path.join(self.dir, "events_{}_{}_{}.json".format(self.country_ISO, self.date_from, self.date_to))

    def download(self):
        """
        Downloads all the ACLED events for a given country and date range
        Group the events by unique pair of coordinates and calculate:
            - The number of unique events
            - The sum of fatalities
            - The number of events of type Violence against civilians
        Save the resulting geojson
        """
        if os.path.exists(self.path):
            print('INFO: ACLED data already downloaded')
            return

        url = "https://api.acleddata.com/acled/read.csv"
        parameters = {"limit": 0, "iso3": self.country_ISO, "fields": "iso|event_type|fatalities|latitude|longitude"}
        parameters["event_date"] = "{" + self.date_from + "|" + self.date_to + "}"
        parameters["event_date_where"] = "BETWEEN"
        response = requests.get(url, params=parameters)

        data = response.content
        data = pd.read_csv(io.StringIO(data.decode('utf-8')))

        print("INFO: Downloaded ", len(data), "ACLED events")

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

    def featurize(self, longitudes, latitudes, property, buffer=5000):
        gdf = gpd.read_file(self.path)
        features = []
        for lat, lon in zip(latitudes, longitudes):
            density = self.__sum_within_bbox(lat, lon, buffer, gdf, property)
            features.append(density)
        return features
