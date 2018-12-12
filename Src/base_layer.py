# -*- coding: utf-8 -*- #
import numpy as np
import gdal


class BaseLayer:
    """ Class that handles the geometries and data.

    We use a raster as input to define the geo-spatial attributes of our data.

    Attributes:
        path_to_raster (str): path to existing raster file.
        x_size (float): size of a pixel in degrees.
        top_left_x_coords (array): longitudes for the top left corner of each pixel.
        self.top_left_y_coords (array): latitudes for the top left corner of each pixel.
        self.centroid_x_coords (array): longitudes for the center of each pixel.
        self.centroid_y_coords (array): latitudes for the center of each pixel.
        self.bands_data (array): values in the raster (for example population density.
        lon, lat (list): lists of the survey's cooridnates.
        i, j (list): lists of the raster's indices.
    """

    def __init__(self, base_raster_file, lon=None, lat=None):
        """
        Args:
            base_raster_file (str): path to the .tif raster to use.
            lon (list): list of longitudes of the survey.
            lat (list): list of latitudes of the survey.
        """
        self.path_to_raster = base_raster_file
        self.path_agg_raster = "../tmp/local_raster.tif"

        self.x_size, \
        self.top_left_x_coords, \
        self.top_left_y_coords, \
        self.centroid_x_coords, \
        self.centroid_y_coords, \
        self.bands_data = self._read_raster(self.path_to_raster)

        if lon is not None:
            self.lon, self.lat = lon, lat
            self.i, self.j = self.get_gridcoordinates(lon, lat)

    def get_gridcoordinates(self, lon, lat):
        """
        takes lon lat gps coordinates and returns the raster indexes.
        Args:
            lon (list): longitudes
            lat (list): longitudes
        Returns:
             i and j lists
        """
        list_i = []
        list_j = []
        for x, y in zip(lon, lat):
            try:
                list_i.append(np.where(self.top_left_x_coords < x)[0][-1])
                list_j.append(np.where(self.top_left_y_coords > y)[0][-1])

            except IndexError:
                print("Coordinates {},{} out of Country bounds".format(x, y))

        return (list_i, list_j)

    def get_gpscoordinates(self, list_i, list_j):
        """ given a set of i and j it returns the lists on longitude and latitude from the grid.
        Arge:
            list_i: list of i (raster/grid references)
            list_j: list of j (raster/grid references)
        :return: two lists containing the gps coordinates corresponding to i and j.
        """

        lon = []
        lat = []
        # for all i and j
        for i, j in zip(list_i, list_j):
            lon.append(self.centroid_x_coords[i])
            lat.append(self.centroid_y_coords[j])

        return (lon, lat)

    def aggregate(self, scale):
        """
        Downsample (upscale) a raster by a given factor and replace no_data value with 0.
        Args:
            scale: The scale (integer) by which the raster in upsampeld.
        Returns:
            Save the output raster to disk.
        # https://github.com/pasquierjb/GIS_RS_utils/blob/master/aggregate_results.py
        """
        import georasters as gr
        input_gr = gr.from_file(self.path_to_raster)

        # No data values are replaced with 0 to prevent summing them in each block.
        input_gr.raster.data[input_gr.raster.data.astype(np.float32) == np.float32(input_gr.nodata_value)] = 0
        input_gr.nodata_value = 0

        output_gr = input_gr.aggregate(block_size=(scale, scale))

        output_gr.to_tiff(self.path_agg_raster.replace(".tif", ""))

        return BaseLayer(self.path_agg_raster, self.lon, self.lat)


    @staticmethod
    def _read_raster(raster_file):
        """ Given the path to a raster file, get the pixel size, pixel location, and pixel value.

        Args:
            raster_file (string): Path to the raster file

        Returns:
            x_size (float): Pixel size
            top_left_x_coords (numpy.ndarray), shape: (number of columns,): Longitude of the top-left point in each pixel
            top_left_y_coords (numpy.ndarray), shape: (number of rows,): Latitude of the top-left point in each pixel
            centroid_x_coords (numpy.ndarray), shape: (number of columns,): Longitude of the centroid in each pixel
            centroid_y_coords (numpy.ndarray), shape: (number of rows,): Latitude of the centroid in each pixel
            bands_data (numpy.ndarray), shape: (number of rows, number of columns, 1): Pixel value
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
        #x_size = 1.0 / int(round(1 / float(x_size)))
        y_size = - x_size
        y_index = np.arange(bands_data.shape[0])
        x_index = np.arange(bands_data.shape[1])
        top_left_x_coords = upper_left_x + x_index * x_size
        top_left_y_coords = upper_left_y + y_index * y_size
        # Add half of the cell size to get the centroid of the cell
        centroid_x_coords = top_left_x_coords + (x_size / 2)
        centroid_y_coords = top_left_y_coords + (y_size / 2)

        return x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data