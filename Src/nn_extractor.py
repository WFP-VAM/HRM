import tensorflow as tf
from tensorflow.python.keras.models import load_model
from pandas import DataFrame
from os import path
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(path.join("..","Src"))


class NNExtractor:
    """
    Class
    -----
    Handles the feature extraction from a pre-trained NN.
    """
    def __init__(self, id, sat, image_dir, model_type, step, GRID):
        """
        Initializes the NNExtractor object where the model to be used is defined.
        :param config: the config file
        """
        self.id = id
        self.sat = sat
        self.model_type = model_type
        self.image_dir = image_dir
        self.step = step
        self.GRID = GRID

        if self.model_type == 'Google':
            print("INFO: loading JB's crappy model for Google Images ...")  # TODO: JB load your model here
            self.net = load_model('../Models/nightGoo.h5', compile=False)
            self.net.layers.pop()
            self.net.layers.pop()
            self.net.layers.pop()
            self.net.layers.pop()
            x = tf.keras.layers.GlobalAveragePooling2D(name='output_maxpool')(self.net.layers[-1].output)
            self.net = tf.keras.models.Model(inputs=self.net.input, outputs=x)

            # from tensorflow.python.keras.applications.vgg16 import VGG16
            # self.net = VGG16(weights='imagenet', include_top=False, pooling='avg')
            # self.net.load_weights('../Models/weights/nigeria-fine-tuned.20170910-160027.lr-1e-07.h5', by_name=True)

        elif self.model_type == 'Sentinel':
            print("INFO: loading model for Sentinel images.")
            self.net = load_model('../Models/nightSent.h5', compile=False)
            self.net.layers.pop()
            self.net.layers.pop()
            self.net.layers.pop()
            self.net.layers.pop()
            x = tf.keras.layers.GlobalAveragePooling2D(name='output_maxpool')(self.net.layers[-1].output)
            self.net = tf.keras.models.Model(inputs=self.net.input, outputs=x)

    def __average_features_dir(self, i, j, provider, start_date, end_date):
        """
        Private function that takes the average of the features computed for all the images in the cluster into one feature.
        :param image_dir: string with path to the folder with images for one tile.
        :return: a list with the averages for each feature extracted from the images in the tile.
        """
        batch_list = []
        c = 0
        for a in range(-self.step, 1 + self.step):
            for b in range(-self.step, 1 + self.step):
                k = i + a
                m = j + b

            try:
                lon = np.round(self.GRID.centroid_x_coords[k], 5)
                lat = np.round(self.GRID.centroid_y_coords[m], 5)
            except IndexError as e:
                lon = np.round(self.GRID.centroid_x_coords[i], 5)
                lat = np.round(self.GRID.centroid_y_coords[j], 5)
                print("Index Error with: ", self.GRID.centroid_x_coords[i], self.GRID.centroid_y_coords[j], "steps: ", a, b)

                if provider == 'Sentinel':
                    file_name = str(lon) + '_' + str(lat) + "_" + str(start_date) + "_" + str(end_date) + '.jpg'
                else:
                    file_name = str(lon) + '_' + str(lat) + '_' + str(16) + '.jpg'

                img_path = os.path.join(self.image_dir, file_name)

                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(400, 400))
                image_preprocess = tf.keras.preprocessing.image.img_to_array(img)
                image_preprocess = np.expand_dims(image_preprocess, axis=0)
                image_preprocess = np.divide(image_preprocess, 255.)

                batch_list.append(image_preprocess)

                c += 1

        features = self.net.predict(np.array(batch_list).reshape(c, 400, 400, 3))
        avg_features = np.mean(features, axis=0)  # take the mean

        return avg_features

    def extract_features(self, list_i, list_j, provider, start_date="2016-01-01", end_date="2017-01-01", pipeline="evaluation"):
        """
        Loops over the folders (tiles) and collects the features.
        :return:
        """
        final = DataFrame([])
        cnt = 0
        total = len(list_i)

        for i, j in zip(list_i, list_j):

            name = str(i) + '_' + str(j)
            cnt += 1

            if cnt % 10 == 0:
                print("Feature extraction : {} tiles out of {}".format(cnt, total), end='\r')

            final[name] = self.__average_features_dir(i, j, provider, start_date, end_date)

        # normalize features
        final = self.scoring_postprocess(final)

        return final

    @staticmethod
    def scoring_postprocess(features):

        features = features.transpose().reset_index()

        # reduce dimensionality
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        out = pd.DataFrame(pca.fit_transform(features.drop('index', axis=1)))

        # normalize the features
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        out = pd.DataFrame(scaler.fit_transform(out))
        out['index'] = features['index']
        # retrieve i and j
        out["i"] = out["index"].apply(lambda x: x.split('_')[0])
        out["j"] = out["index"].apply(lambda x: x.split('_')[1])
        out["i"] = pd.to_numeric(out["i"])
        out["j"] = pd.to_numeric(out["j"])

        return out
