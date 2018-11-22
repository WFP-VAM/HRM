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

    # def distance_to_nearest2(self, df, points_gdf, lat_col="gpsLatitude", lon_col="gpsLongitude"):
    #     '''
    #     Ditance for a point in a pandas dataframe to the nearest point in a geodataframe.
    #     '''
    #     # To do: Use a spatial index for faster computation
    #     from shapely.ops import nearest_points
    #     from shapely.geometry import Point
    #     geom_union = points_gdf.unary_union
    #     point = Point(df[lon_col], df[lat_col])
    #     nearest_point = nearest_points(point, geom_union)[1]
    #     distance = nearest_point.distance(point)
    #     return distance

    def gpd_to_tree(self, points_gdf):
        import scipy.spatial as spatial
        gps = []
        for x in points_gdf["geometry"]:
            gps.append(x.coords[:][0])
        point_tree = spatial.cKDTree(gps)
        return point_tree

    def distance_to_nearest(self, latitudes, longitudes, point_tree):
        """ Ditance between a point in a pandas dataframe and the nearest point in a scipy kd-tree.
        """
        distances = []
        for lon, lat in zip(longitudes, latitudes):
            distances.append(point_tree.query([lon, lat], k=1)[0])
        return distances

    def density(self, df, points_gdf, distance=5000, lat_col="gpsLatitude", lon_col="gpsLongitude"):
        '''
        Number of points in a geodataframe within a box around a point in a dataframe.
        '''
        # To do: Use a spatial index for faster computation
        from osmnx.core import bbox_from_point
        from shapely.geometry import Polygon
        point = (df[lon_col], df[lat_col])
        north, south, east, west = bbox_from_point(point, distance)
        point1 = (east, south)
        point2 = (west, south)
        point3 = (west, north)
        point4 = (east, north)
        poly = Polygon([point1, point2, point3, point4])
        n = points_gdf.within(poly).sum()
        return n
