import os
import yaml
from img_utils import get_cell_idx
import gdal
import numpy as np
import sentinel_utils


with open('../private_config.yml', 'r') as cfgfile:
    tokens = yaml.load(cfgfile)


class RasterGrid:
    """
    Class
    -----
    Handles the raster grid, and related functionalities like downloading the relevant satellite images
    from Google Static API.
    """

    def __init__(self, raster):

        self.x_size, \
        self.top_left_x_coords, \
        self.top_left_y_coords, \
        self.centroid_x_coords, \
        self.centroid_y_coords, \
        self.bands_data = self._read_raster(raster)
        self.url = None
        self.image_dir = None

    def get_gridcoordinates(self, dataset, lon_col='gpsLongitude', lat_col='gpsLatitude'):
        """
        takes a dataset with gps coordinates and returns the raster indexes.
        :param dataset: pandas dataframe
        :param lon_col: longitude columns (string)
        :param lat_col: latitude columns (string)
        :return: i and j lists
        """
        list_i=[]
        list_j=[]
        for index, row in dataset.iterrows():
            try:
                i, j = get_cell_idx(row[lon_col], row[lat_col], self.top_left_x_coords, self.top_left_y_coords)
            except IndexError:
                print("Corrdinates {},{} out of Country bounds".format(row[lon_col],row[lat_col]))
            list_i.append(i)
            list_j.append(j)

        return (list_i, list_j)

    def get_gridcoordinates2(self, list_lat, list_lon):
        """
        takes gps coordinates and returns the raster indexes.
        :param list_lon: lsit with longitudes
        :param list_lat: list with latitudes
        :return: i and j lists
        """
        list_i=[]
        list_j=[]
        for j, i in zip(list_lat, list_lon):
            ii, jj = get_cell_idx(i, j, self.top_left_x_coords, self.top_left_y_coords)
            list_i.append(ii)
            list_j.append(jj)
        return (list_i, list_j)

    def get_gpscoordinates(self, list_i, list_j):
        """
        given a set of i and j it returns the lists on longitude and latitude.
        :param list_i: list of i (raster/grid references)
        :param list_j: list of j (raster/grid references)
        :param step: the steps around the cluster
        :return: two lists containing the gps coordinates corresponding to i and j.
        """

        lon = []
        lat = []
        # for all i and j
        for i, j in zip(list_i, list_j):
            lon.append(self.centroid_x_coords[i])
            lat.append(self.centroid_y_coords[j])

        return(lon, lat)

    def _read_raster(self, raster_file):
        """
        Function
        --------
        read_raster

        Given a raster file, get the pixel size, pixel location, and pixel value

        Parameters
        ----------
        raster_file : string
            Path to the raster file

        Returns
        -------
        x_size : float
            Pixel size
        top_left_x_coords : numpy.ndarray  shape: (number of columns,)
            Longitude of the top-left point in each pixel
        top_left_y_coords : numpy.ndarray  shape: (number of rows,)
            Latitude of the top-left point in each pixel
        centroid_x_coords : numpy.ndarray  shape: (number of columns,)
            Longitude of the centroid in each pixel
        centroid_y_coords : numpy.ndarray  shape: (number of rows,)
            Latitude of the centroid in each pixel
        bands_data : numpy.ndarray  shape: (number of rows, number of columns, 1)
            Pixel value
        """

        gdal.UseExceptions()

        raster_dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
        # get project coordination
        bands_data = []
        # Loop through all raster bands
        for b in range(1, raster_dataset.RasterCount + 1):
            band = raster_dataset.GetRasterBand(b)
            bands_data.append(band.ReadAsArray())
        bands_data = np.dstack(bands_data)

        # Get the metadata of the raster
        geo_transform = raster_dataset.GetGeoTransform()
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = geo_transform

        # Get location of each pixel
        x_size = 1.0 / int(round(1 / float(x_size)))
        y_size = - x_size
        y_index = np.arange(bands_data.shape[0])
        x_index = np.arange(bands_data.shape[1])
        top_left_x_coords = upper_left_x + x_index * x_size
        top_left_y_coords = upper_left_y + y_index * y_size
        # Add half of the cell size to get the centroid of the cell
        centroid_x_coords = top_left_x_coords + (x_size / 2)
        centroid_y_coords = top_left_y_coords + (y_size / 2)

        return x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data

    def download_images(self, list_i, list_j, step, provider, start_date="2017-01-01", end_date="2018-01-01"):
        """
        Function
        --------
        given the list of tiles, it downloads the corresponding satellite images.

        Parameters
        ----------
        list_i, list_j: the list of tiles that need an image.
        step: int, varies depending on provider.
        """

        cnt = 0
        total = len(list_i)*(2*step+1)**2
        # compose the file path and make the directory of not there
        self.image_dir = os.path.join("../Data", "Satellite", provider + '/')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        for i, j in zip(list_i, list_j):
            for a in range(-step, 1+step):
                for b in range(-step, 1+step):

                    print("INFO: {} images downloaded out of {}".format(cnt,total),end='\r')
                    cnt += 1

                    lon = np.round(self.centroid_x_coords[i + a], 5)
                    lat = np.round(self.centroid_y_coords[j + b], 5)

                    if (provider == 'Sentinel') or (provider == 'Sentinel_maxNDVI'):
                        file_name = str(lon) + '_' + str(lat) + "_" + str(start_date)+"_"+str(end_date) + '.jpg'
                    else:
                        file_name = str(lon) + '_' + str(lat) + '_' + str(16) + '.jpg'

                    if os.path.exists(self.image_dir + file_name):
                        print("INFO: {} already downloaded".format(file_name))
                    else:
                        url = self._produce_url(lon, lat, provider, start_date, end_date)
                        self._save_img(url, self.image_dir, file_name, provider)

    def _produce_url(self, lon, lat, provider, start_date, end_date):
        """wrapper for the creation of the url depending on the engine."""

        if provider == 'Google':
            return('https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' +
                    str(lon) + '&zoom=16&size=400x500&maptype=satellite&key=' + tokens['Google'])

        elif provider == 'Bing':
            imagery_set = "Aerial"
            zoom_level = "16"
            map_size = "400,500"
            center_point = str(lat) + "," + str(lon)

            return("http://dev.virtualearth.net/REST/v1/Imagery/Map/" + imagery_set + "/" + center_point +
                    "/" + zoom_level + "?mapSize=" + map_size + "&key=" + tokens['Bing'])

        elif provider == 'Sentinel':
            d=5000
            geojson=sentinel_utils.squaretogeojson(lon,lat,d)
            url=sentinel_utils.gee_url(geojson,str(start_date),str(end_date))
            return url

        elif provider == 'Sentinel_maxNDVI':
            d=5000
            geojson=sentinel_utils.squaretogeojson(lon,lat,d)
            url=sentinel_utils.gee_maxNDBImaxNDVImaxNDWI_url(geojson,str(start_date),str(end_date))
            return url

        else:
            print("ERROR: Wrong API {}".format(provider))

    def _save_img(self, url, file_path, file_name, provider):
        """
        Function
        --------
        this private method is used by the "download_images" to call the Google API
        for every pair of coordinates, to parse the request and to save the image to the folder.
        """
        import urllib.error
        from scipy import misc
        from scipy.misc.pilutil import imread
        from io import BytesIO

        ur = urllib.request.urlopen(url).read()
        buffer = BytesIO(ur)

        if (provider == 'Sentinel') or (provider == 'Sentinel_maxNDVI'):
            gee_tif = sentinel_utils.download_and_unzip(buffer, 3, 6, file_path)
            try:
                sentinel_utils.rgbtiffstojpg(gee_tif, file_path, file_name)
            except:
                print("GEE error with :{}".format(file_name))

        else:
            image = imread(buffer, mode='RGB')
            if (image[:, :, 0] == 245).sum() >= 100000: #Gray image in Bing
                print("No image in Bing API")
            else:
                print('file path: ', file_path)
                print('file name: ', file_name)

                misc.imsave(file_path + file_name, image[50:450, :, :])
