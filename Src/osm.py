class OSM_extractor:

    def __init__(self, minlon, minlat, maxlon, maxlat):
        self.minlon, self.minlat, self.maxlon, self.maxlat = minlon, minlat, maxlon, maxlat

    def download(self, tag_key='amenity', tag_value='school'):
        '''
        Get the json of coordinates (or the number of items) within a bbox for a specific osm tag.
        https://taginfo.openstreetmap.org/
        https://wiki.openstreetmap.org/wiki/Map_Features#Building
        '''
        from osmnx.core import overpass_request
        from shapely.geometry import Point
        import geopandas as gpd
        import os

        if os.path.exists("../Data/Geofiles/OSM/location_{}_{}_{}_{}_{}_{}.json".format(tag_key, tag_value, self.minlat, self.maxlat, self.minlon, self.maxlon)):
            print('INFO: OSM data for {} = {} already downloaded'.format(tag_key, tag_value))
            gdf = gpd.read_file("../Data/Geofiles/OSM/location_{}_{}_{}_{}_{}_{}.json".format(tag_key, tag_value, self.minlat, self.maxlat, self.minlon, self.maxlon))
            return gdf

        query_osm = ('[out:json][maxsize:2000000000];'
                     '('
                     'node["{tag_key}"="{tag_value}"]({minlat:.8f},{minlon:.8f},{maxlat:.8f},{maxlon:.8f});'
                     'way["{tag_key}"="{tag_value}"]({minlat:.8f},{minlon:.8f},{maxlat:.8f},{maxlon:.8f});'
                     'relation["{tag_key}"="{tag_value}"]({minlat:.8f},{minlon:.8f},{maxlat:.8f},{maxlon:.8f});'
                     ');(._;>;);out center;'
                     ).format(minlat=self.minlat, maxlat=self.maxlat, minlon=self.minlon, maxlon=self.maxlon, tag_key=tag_key, tag_value=tag_value)

        print('INFO: Downloading OSM data for {} = {}'.format(tag_key, tag_value))
        # overpass_request is already saving json to a cache folder
        response_json = overpass_request(data={'data': query_osm}, timeout=10000, error_pause_duration=None)
        print('INFO: OSM data for {} = {} downloaded. N lines: '.format(tag_key, tag_value, len(response_json)))
        points = []
        for result in response_json['elements']:
            if 'type' in result and result['type'] == 'node':
                p = Point([(result['lon'], result['lat'])])
                point = {'geometry': p}
                points.append(point)
            if 'type' in result and result['type'] == 'way':
                p = Point([(result['center']['lon'], result['center']['lat'])])
                point = {'geometry': p}
                points.append(point)
        gdf = gpd.GeoDataFrame(points)
        gdf.crs = {'init': 'epsg:4326'}
        gdf.to_file("../Data/Geofiles/OSM/location_{}_{}_{}_{}_{}_{}.json".format(tag_key, tag_value, self.minlat, self.maxlat, self.minlon, self.maxlon), driver='GeoJSON')
        gdf.to_file("../Data/Geofiles/OSM/location_{}_{}_{}_{}_{}_{}.shp".format(tag_key, tag_value, self.minlat, self.maxlat, self.minlon, self.maxlon), driver='ESRI Shapefile')

        return gdf

    def distance_to_nearest(self, latitudes, longitudes, gdf):
        """ Ditance between a point in a pandas dataframe and the nearest point in a scipy kd-tree.
        """
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        gdf_lats = gdf["geometry"].y
        gfd_lons = gdf["geometry"].x
        X = np.array([gdf_lats, gfd_lons]).T
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(X)
        features = []
        for lat, lon in zip(latitudes, longitudes):
            X = np.array([[lat, lon]])
            a, _ = nbrs.kneighbors(X)
            features.append(a[[0]][0][0])
        return features
