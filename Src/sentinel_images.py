# -*- coding: utf-8 -*-#
from data_source import DataSource
import os
import yaml
from utils import retry
from urllib.request import urlopen
from utils import squaretogeojson, gee_url
from io import BytesIO
import tensorflow as tf
import numpy as np

with open('../private_config.yml', 'r') as cfgfile:
    tokens = yaml.load(cfgfile)


class SentinelImages(DataSource):
    """overloading the DataSource class."""

    def __init__(self, directory):
        DataSource.__init__(self, directory)

        """ Overload the directory path. """
        self.directory = os.path.join(self.directory, 'Sentinel/')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        """ loads the model. """
        print("INFO: loading model for Sentinel images.")
        self.net = tf.keras.models.load_model('../Models/nightSent.h5', compile=False)
        self.net.layers.pop()
        self.net.layers.pop()
        self.net.layers.pop()
        self.net.layers.pop()
        x = tf.keras.layers.GlobalAveragePooling2D(name='output_maxpool')(self.net.layers[-1].output)
        self.net = tf.keras.models.Model(inputs=self.net.input, outputs=x)

    def download(self, lon, lat, start_date=None, end_date=None, img_size=5000):
        """
        given the list of coordinates, it downloads the corresponding satellite images.

        Args:
            lon (list): list of longitudes.
            lat (list): list of latitudes.
            start_date (str): take images from ...
            end_date (str): take images to ...
            img_size (int): size in meters of the image.
        """

        @retry(Exception, tries=4)
        def _urlopen_with_retry(url):
            return urlopen(url).read()

        _cnt, _total = 0, len(lon)  # counter and total number of images.

        for i, j in zip(lon, lat):

            print("INFO: {} images downloaded out of {}".format(_cnt, _total), end='\r')
            _cnt += 1

            file_name = str(i) + '_' + str(j) + "_" + str(start_date) \
                        + "_" + str(end_date) + "_" + str(img_size) + '.jpg'

            if os.path.exists(self.directory + file_name):
                print("INFO: {} already downloaded".format(file_name), end='\r')
            else:
                geojson = squaretogeojson(i, j, img_size)
                url = gee_url(geojson, str(start_date), str(end_date))
                buffer = BytesIO(_urlopen_with_retry(url))

                gee_tif = self._download_and_unzip(buffer, 3, 6, self.directory)

                self._rgbtiffstojpg(gee_tif, self.directory, file_name)

    def featurize(self, lon, lat, start_date=None, end_date=None, img_size=5000):
        """ Given a lon lat pair, it extract the features from the image (if there) using the NN.

        Args:
            lon (list): list of longitudes.
            lat (lsit): list of latitudes.
            start_date (str): take images from ...
            end_date (str): take images to ...

        Returns:
            covariates for the coordinates pair.
        """
        _cnt, _total = 0, len(lon)  # counter and total number of images.

        features = []
        for i, j in zip(lon, lat):
            _cnt += 1

            file_name = str(i) + '_' + str(j) + "_" + str(start_date) + "_" + str(end_date) + "_" + str(img_size) + '.jpg'
            img_path = os.path.join(self.directory, file_name)

            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(400, 400))
            image_preprocess = tf.keras.preprocessing.image.img_to_array(img)
            image_preprocess = np.expand_dims(image_preprocess, axis=0)
            image_preprocess = np.divide(image_preprocess, 255.)

            features.append(self.net.predict(np.array(image_preprocess).reshape(1, 400, 400, 3)))

            if _cnt % 10 == 0: print("Feature extraction : {} tiles out of {}".format(_cnt, _total), end='\r')

        #  TODO: save transforms for predicting.
        # reduce dimensionality
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        out = pca.fit_transform(np.array(features).reshape(len(features), 256))

        # normalize the features
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        out = scaler.fit_transform(out)

        return out

    @staticmethod
    def _download_and_unzip(buffer, a, b, path):
        unzipped = []
        import zipfile
        try:
            zip_file = zipfile.ZipFile(buffer)
        except zipfile.BadZipFile:  # Often happens with GEE API
            print("bad_zip")
            return None
        files = zip_file.namelist()
        for i in range(a, b):
            zip_file.extract(files[i], path + "/tiff/")
            # print("{} downloaded and unzippped".format(files[i]))
            unzipped.append(files[i])
        return unzipped

    @staticmethod
    def _rgbtiffstojpg(files, path, name):
        """
        files: a list of files ordered as follow. 0: Blue Band 1: Green Band 2: Red Band
        path: the path to look for the tiff files and to save the jpg
        """
        import scipy.misc as sm
        import gdal
        import numpy as np
        b2_link = gdal.Open(path + "/tiff/" + files[0])
        b3_link = gdal.Open(path + "/tiff/" + files[1])
        b4_link = gdal.Open(path + "/tiff/" + files[2])

        # call the norm function on each band as array converted to float
        def norm(band):
            band_min, band_max = band.min(), band.max()
            return (band - band_min) / (band_max - band_min)

        b2 = norm(b2_link.ReadAsArray().astype(np.float))
        b3 = norm(b3_link.ReadAsArray().astype(np.float))
        b4 = norm(b4_link.ReadAsArray().astype(np.float))

        # Create RGB
        rgb = np.dstack((b4, b3, b2))
        del b2, b3, b4
        sm.toimage(rgb, cmin=np.percentile(rgb, 2), cmax=np.percentile(rgb, 98)).save(path + name)

# unit-test
def test_SentinelImages():
    simages = SentinelImages('test/')
    simages.download([12.407305, 6.864997], [41.821816, 45.832565], start_date='2017-01-01', end_date='2018-01-01')
    f = simages.featurize([12.407305, 6.864997], [41.821816, 45.832565], start_date='2017-01-01', end_date='2018-01-01')