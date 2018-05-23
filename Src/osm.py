
class OSM_extractor:

    def __init__(self, df):
        self.minlat, self.maxlat, self.minlon, self.maxlon = self.__boundaries(df)

    def __boundaries(self, df, buffer=0.05, lat_col="gpsLatitude", lon_col="gpsLongitude"):
        '''
        Get GPS coordinates of the boundary box of a DataFrame and add some buffer around it.
        '''
        minlat = df["gpsLatitude"].min()
        maxlat = df["gpsLatitude"].max()
        minlon = df["gpsLongitude"].min()
        maxlon = df["gpsLongitude"].max()

        lat_buffer = (maxlat - minlat) * buffer
        lon_buffer = (maxlon - minlon) * buffer

        minlat = minlat - lat_buffer
        maxlat = maxlat + lat_buffer
        minlon = minlon - lon_buffer
        maxlon = maxlon + lon_buffer

        return minlat, maxlat, minlon, maxlon

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

        if os.path.exists("../Data/Geofiles/OSM/location_{}_{}.json".format(tag_key, tag_value)):
            print('OSM data for {} = {} already downloaded'.format(tag_key, tag_value))
            gdf = gpd.read_file("../Data/Geofiles/OSM/location_{}_{}.json".format(tag_key, tag_value))
            return gdf
        query_osm = ('[out:json][timeout:25];'
                     '('
                     'node["{tag_key}"="{tag_value}"]({minlat:.8f},{minlon:.8f},{maxlat:.8f},{maxlon:.8f});'
                     'way["{tag_key}"="{tag_value}"]({minlat:.8f},{minlon:.8f},{maxlat:.8f},{maxlon:.8f});'
                     'relation["{tag_key}"="{tag_value}"]({minlat:.8f},{minlon:.8f},{maxlat:.8f},{maxlon:.8f});'
                     ');(._;>;);out center;'
                     ).format(minlat=self.minlat, maxlat=self.maxlat, minlon=self.minlon, maxlon=self.maxlon, tag_key=tag_key, tag_value=tag_value)
        print('Downloading OSM data for {} = {}'.format(tag_key, tag_value))
        # overpass_request is already saving json to a cache folder
        response_json = overpass_request(data={'data': query_osm}, timeout=600, error_pause_duration=None)

        points = []
        for result in response_json['elements']:
            if 'type' in result and result['type'] == 'node':
                p = Point([(result['lat'], result['lon'])])
                point = {'geometry': p}
                points.append(point)
            if 'type' in result and result['type'] == 'way':
                p = Point([(result['center']['lat'], result['center']['lon'])])
                point = {'geometry': p}
                points.append(point)
        gdf = gpd.GeoDataFrame(points)
        gdf.crs = {'init': 'epsg:4326'}
        gdf.to_file("../Data/Geofiles/OSM/location_{}_{}.json".format(tag_key, tag_value), driver='GeoJSON')
        return gdf

    def distance_to_nearest(self, df, points_gdf, lat_col="gpsLatitude", lon_col="gpsLongitude"):
        '''
        Ditance for a point in a pandas dataframe to the nearest point in a geodataframe.
        '''
        # To do: Use a spatial index for faster computation
        from shapely.ops import nearest_points
        from shapely.geometry import Point
        geom_union = points_gdf.unary_union
        point = Point(df[lat_col], df[lon_col])
        nearest_point = nearest_points(point, geom_union)[1]
        distance = nearest_point.distance(point)
        return distance

    def density(self, df, points_gdf, distance=5000, lat_col="gpsLatitude", lon_col="gpsLongitude"):
        '''
        Number of points in a geodataframe within a box around a point in a dataframe.
        '''
        # To do: Use a spatial index for faster computation
        from osmnx.core import bbox_from_point
        from shapely.geometry import Polygon
        point = (df[lat_col], df[lon_col])
        north, south, east, west = bbox_from_point(point, distance)
        point1 = (south, east)
        point2 = (south, west)
        point3 = (north, west)
        point4 = (north, east)
        poly = Polygon([point1, point2, point3, point4])
        n = points_gdf.within(poly).sum()
        return n
