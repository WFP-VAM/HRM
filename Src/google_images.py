# -*- coding: utf-8 -*-#
from data_source import DataSource
import os
from utils import retry, s3_download
from urllib.request import urlopen
from io import BytesIO
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

# vgg16 performs better in predicting nightlights but produces worse scoring features
MODEL = 'google_njean.h5' # 'google_simple.h5'
# TODO: we can allow a parameter in the config for the model to use, as long as the layer is called 'features'
LAYER = 'conv7' #'dense'
IMG_SIZE = 400
# Google images parameters.
ZOOM_LEVEL = 16
map_size = "400x500"
imagery_set = "satellite"


class GoogleImages(DataSource):
    """overloading the DataSource class."""

    def __init__(self, directory):
        DataSource.__init__(self, directory)

        """ Overload the directory path. """
        self.directory = os.path.join(self.directory, 'Google/')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        """ loads the model. """
        if os.path.exists('../Models/{}'.format(MODEL)) is False:
            print("INFO: downloading model. ")
            with requests.get('https://hrm-models.s3.eu-central-1.amazonaws.com/{}'.format(MODEL), stream=True) as r:
                if r.status_code == 200:
                    with open('../Models/{}'.format(MODEL), 'wb') as f:
                        f.write(r.content)
                else:
                    raise Exception('error reading model form S3. Error code ', r.status_code)

        print("INFO: loading model for Google Images ...")
        self.net = tf.keras.models.load_model('../Models/{}'.format(MODEL), compile=False)
        self.net = tf.keras.models.Model(
            inputs=self.net.input,
            outputs=self.net.get_layer(LAYER).output)

    def download(self, lon, lat, step=False):
        """ given the list of coordinates, it downloads the corresponding satellite images.

        Notes:
            Google Maps Static API takes {latitude,longitude}

        Args:
            lon (list): list of longitudes.
            lat (list): list of latitudes.
            step (bool): if you want to add buffer images. SMore accurate but slow.
        """
        if step:
            print('INFO: adding steps to coordinates set.')
            lon, lat = self.add_steps(lon, lat)

        @retry(Exception, tries=4)
        def _urlopen_with_retry(url):
            return urlopen(url).read()

        _cnt, _total = 0, len(lon)  # counter and total number of images.

        for i, j in zip(lon, lat):

            print("INFO: {} images downloaded out of {}".format(_cnt, _total), end='\r')
            _cnt += 1

            file_name = str(i) + '_' + str(j) + '_' + str(ZOOM_LEVEL) + '.jpg'

            if os.path.exists(self.directory + file_name):
                print("INFO: {} already downloaded".format(file_name), end='\r')
            else:
                center_point = str(j) + "," + str(i)

                url = """https://maps.googleapis.com/maps/api/staticmap?center={}&zoom={}&size={}&maptype={}&key={}""".\
                    format(center_point, ZOOM_LEVEL, map_size, imagery_set, os.environ['Google_key'])

                buffer = BytesIO(_urlopen_with_retry(url))

                image = np.asarray(Image.open(buffer).convert('RGB'))#

                print('file path: ', os.path.join(self.directory, file_name))
                Image.fromarray(image[50:450, :, :]).save(os.path.join(self.directory, file_name))

    def featurize(self, lon, lat, step=0):
        """ Given lon lat lists, it extract the features from the image (if there) using the NN.

        Args:
            lon (list): list of longitudes.
            lat (list): list of latitudes.
            step (bool): if you want to add buffer images (9 in total). More accurate but slow.

        Returns:
            covariates for the coordinates pair.
        """
        if step:
            print('INFO: adding steps to coordinates set.')
            lon, lat = self.add_steps(lon, lat)

        _cnt, _total = 0, len(lon)  # counter and total number of images.

        features = []
        for i, j in zip(lon, lat):
            _cnt += 1

            file_name = str(i) + '_' + str(j) + '_' + str(ZOOM_LEVEL) + '.jpg'
            img_path = os.path.join(self.directory, file_name)

            image = Image.open(img_path, 'r')
            image = image.crop((  # crop center
                int(image.size[0] / 2 - IMG_SIZE / 2),
                int(image.size[1] / 2 - IMG_SIZE / 2),
                int(image.size[0] / 2 + IMG_SIZE / 2),
                int(image.size[1] / 2 + IMG_SIZE / 2)
            ))
            image = np.array(image)/ 255.
            features.append(self.net.predict(np.array(image).reshape(1, IMG_SIZE, IMG_SIZE, 3)))

            if _cnt % 10 == 0: print("Feature extraction : {} tiles out of {}".format(_cnt, _total), end='\r')

        if step:
            # take the average for the 9 images around the original lon,lat
            f = np.copy(features)
            features = []
            for c in range(0, int(len(f)/9)):
                lower_bound = c*9
                features.append(np.mean(f[lower_bound:(lower_bound+9)], axis=0))

            features = np.array(features)

        #  TODO: save transforms for predicting.
        # reduce dimensionality
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        out = pca.fit_transform(np.array(features).reshape(len(features), -1))

        return out

    @staticmethod
    def add_steps(lon, lat, step=0.009):
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