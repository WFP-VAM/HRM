# -*- coding: utf-8 -*-#
from data_source import DataSource
import os
import yaml
from utils import retry
from urllib.request import urlopen
from io import BytesIO
from scipy.misc.pilutil import imread, imsave
import tensorflow as tf
import numpy as np

with open('../private_config.yml', 'r') as cfgfile:
    tokens = yaml.load(cfgfile)


class GoogleImages(DataSource):
    """overloading the DataSource class."""

    def __init__(self, directory):
        DataSource.__init__(self, directory)

        """ Overload the directory path. """
        self.directory = os.path.join(self.directory, 'Google/')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        """ loads the model. """
        print("INFO: loading model for Google Images ...")
        self.net = tf.keras.models.load_model('../Models/nightGoo.h5', compile=False)
        self.net.layers.pop()
        self.net.layers.pop()
        self.net.layers.pop()
        self.net.layers.pop()
        x = tf.keras.layers.GlobalAveragePooling2D(name='output_maxpool')(self.net.layers[-1].output)
        self.net = tf.keras.models.Model(inputs=self.net.input, outputs=x)

    def download(self, lon, lat, step=False):
        """ given the list of coordinates, it downloads the corresponding satellite images.

        Notes:
            Google Maps Static API takes {latitude,longitude}

        Args:
            lon (list): list of longitudes.
            lat (lsit): list of latitudes.
            step (bool): if you want to add buffer images. SMore accurate but slow.
        """
        if step:
            print('INFO: adding steps to coordinates set.')
            lon, lat = self.add_steps(lon, lat)

        # Google images parameters.
        zoom_level = 16
        map_size = "400x500"
        imagery_set = "satellite"

        @retry(Exception, tries=4)
        def _urlopen_with_retry(url):
            return urlopen(url).read()

        _cnt, _total = 0, len(lon)  # counter and total number of images.

        for i, j in zip(lon, lat):

            print("INFO: {} images downloaded out of {}".format(_cnt, _total), end='\r')
            _cnt += 1

            file_name = str(i) + '_' + str(j) + '_' + str(zoom_level) + '.jpg'

            if os.path.exists(self.directory + file_name):
                print("INFO: {} already downloaded".format(file_name), end='\r')
            else:
                center_point = str(j) + "," + str(i)

                url = """https://maps.googleapis.com/maps/api/staticmap?center={}&zoom={}&size={}&maptype={}&key={}""".\
                    format(center_point, zoom_level, map_size, imagery_set, tokens['Google'])

                buffer = BytesIO(_urlopen_with_retry(url))

                image = imread(buffer, mode='RGB')
                if (image[:, :, 0] == 245).sum() >= 100000:  # Gray image in Bing
                    print("No image in Bing API", file_name)
                else:
                    print('file path: ', os.path.join(self.directory, file_name))
                    imsave(os.path.join(self.directory, file_name), image[50:450, :, :])

    def featurize(self, lon, lat, step=False):
        """ Given a lon lat pair, it extract the features from the image (if there) using the NN.

        Args:
            lon (list): list of longitudes.
            lat (lsit): list of latitudes.
            step (bool): if you want to add buffer images. SMore accurate but slow.

        Returns:
            covariates for the coordinates pair.
        """
        if step:
            print('INFO: adding steps to coordinates set.')
            lon, lat = self.add_steps(lon, lat)

        _cnt, _total = 0, len(lon) # counter and total number of images.

        features = []
        for i, j in zip(lon, lat):
            _cnt += 1

            file_name = str(i) + '_' + str(j) + '_' + str(16) + '.jpg'
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
    def add_steps(lon, lat, step = 0.009):
        """
        returns the augmented set of coordinates for all the 9 images adjacent to the center.
        0.009 ~ 1km at the equator.
        """
        new_i, new_j = [], []
        for i, j in zip(lon, lat):
            for a in [-step, 0, step]:
                for b in [-step, 0, step]:
                    new_i.append(i + a)
                    new_j.append(j + b)

        return new_i, new_j


# unit-test
def test_GoogleImages():
    gimages = GoogleImages('test/')
    gimages.download([12.407305, 6.864997], [41.821816, 45.832565], step=True)
    f = gimages.featurize([12.407305, 6.864997], [41.821816, 45.832565], step=True)