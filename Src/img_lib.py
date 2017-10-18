import os

class RasterGrid:
    """
    Class
    -----
    Handles the raster grid, and related functionalities like downloading the relevant satellite images
    from Google Static API.
    """

    def __init__(self, raster_file='../Data/Satellite/NightLight/F182013.v4c_web.stable_lights.avg_vis.tif'):
          self.x_size, \
          self.top_left_x_coords, \
          self.top_left_y_coords, \
          self.centroid_x_coords, \
          self.centroid_y_coords, \
          self.bands_data = self.__read_raster(raster_file)
          self.url = None

    def __read_raster(self, raster_file):
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
        import gdal
        import numpy as np
        gdal.UseExceptions()

        raster_dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
        # get project coordination
        bands_data = []
        # Loop through all raster bands
        for b in range(1, raster_dataset.RasterCount + 1):
            band = raster_dataset.GetRasterBand(b)
            bands_data.append(band.ReadAsArray())
            no_data_value = band.GetNoDataValue()
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

    def download_images(self, list_i, list_j, config, api="google",steps_per_tile=0):
        """
        Function
        --------
        given the list of tiles, it downloads the corresponding satellite images.

        Parameters
        ----------
        list_i, list_j: the list of tiles that need an image.
        config: the config file.
        steps_per_tile=0: the number of +\- steps in the x and y to pull the image for. Default 0 so only one image per tile.

        """

        m = 1
        for i, j in zip(list_i, list_j):

            file_path = config['output_image_dir'] + str(i) + '_' + str(j) + '/'
            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            for a in range(-steps_per_tile, steps_per_tile):
                for b in range(-steps_per_tile, steps_per_tile):
                    lon = self.centroid_x_coords[i + a]
                    lat = self.centroid_y_coords[j + b]
                    self.url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' + \
                          str(lon) + '&zoom=16&size=400x500&maptype=satellite&key=' + config['api_token']
                    file_name = str(i + a) + '_' + str(j + b) + '.jpg'
                    self.__save_img(file_path, file_name)
                    if m % 20 == 0:
                        print('pulled {} images ...'.format(m))
                    m += 1


    def __save_img(self, file_path, file_name):
        """
        Function
        --------
        this private method is used by the "download_images" to call the Google API
        for every pair of coordinates, to parse the request and to save the image to the folder.
        """
        import numpy as np
        import urllib
        from scipy import misc
        from scipy.misc.pilutil import imread
        # this lib is different between Python 3x and Python 2x
        try:
            from StringIO import StringIO
        except ImportError:
            from io import BytesIO

        try:
            uropen = urllib.request.urlopen(self.url)
            a = uropen.read()
        except urllib.error.HTTPError as e:
            a = e.read()

        b = BytesIO(a)
        image = imread(b, mode='RGB')
        if np.array_equal(image[:, :10, :], image[:, 10:20, :]):
            print("bad image")
        else:
            misc.imsave(file_path + file_name, image[50:450, :, :])
