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

        from tensorflow.python.keras.applications.vgg16 import VGG16
        from tensorflow.python.keras.applications.resnet50 import ResNet50

        if self.model_type == 'ResNet50':
            print('INFO: loading ResNet50 ...')
            self.net = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        elif self.model_type == 'VGG16':
            print('INFO: loading VGG16 ...')
            self.net = VGG16(weights='imagenet', include_top=False, pooling='avg')
        else:
            print("ERROR: Only ResNet50 and VGG16 implemented so far")

    def load_weights(self, weights_path):
        print('INFO: loading custom weights ...')
        self.net.load_weights(weights_path, by_name=True)

    def __average_features_dir(self, i, j, provider,start_date,end_date):
        """
        Private function that takes the average of the features computed for all the images in the cluster into one feature.
        :param image_dir: string with path to the folder with images for one tile.
        :return: a list with the averages for each feature extracted from the images in the tile.
        """
        import os
        import numpy as np
        import tensorflow as tf
        if self.model_type == 'ResNet50':
            preprocess_input = tf.keras.applications.resnet50.preprocess_input
        elif self.model_type == 'VGG16':
            preprocess_input = tf.keras.applications.vgg16.preprocess_input

        else:
            print('ERROR: only ResNet50 and VGG16 implemented so far')

        batch_list = []
        c = 0
        for a in range(-self.step, 1 + self.step):
            for b in range(-self.step, 1 + self.step):
                k = i + a
                l = j + b

                lon = np.round(self.GRID.centroid_x_coords[k], 5)
                lat = np.round(self.GRID.centroid_y_coords[l], 5)

                if (provider == 'Sentinel') or (provider == 'Sentinel_maxNDVI'):
                    file_name = str(lon) + '_' + str(lat) + "_" + str(start_date) + "_" + str(end_date) + '.jpg'
                else:
                    file_name = str(lon) + '_' + str(lat) + '_' + str(16) + '.jpg'

                img_path = os.path.join(self.image_dir, file_name)

                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(400, 400))
                image_preprocess = tf.keras.preprocessing.image.img_to_array(img)
                image_preprocess = np.expand_dims(image_preprocess, axis=0)
                image_preprocess = preprocess_input(image_preprocess)

                batch_list.append(image_preprocess)

                c += 1

        features = self.net.predict(np.array(batch_list).reshape(c, 400, 400,3))

        avg_features = np.mean(features, axis=0)

        return avg_features

    def extract_features(self, list_i, list_j, provider, start_date="2016-01-01", end_date="2017-01-01", pipeline="evaluation"):
        """
        Loops over the folders (tiles) and collects the features.
        :return:
        """
        from pandas import DataFrame
        from os import path
        import sys
        sys.path.append(path.join("..","Src"))
        from utils import scoring_postprocess

        final = DataFrame([])
        cnt = 0
        total = len(list_i)

        for i, j in zip(list_i, list_j):

            name = str(i) + '_' + str(j)
            cnt += 1

            if cnt%10: print("Feature extraction : {} tiles out of {}".format(cnt, total), end='\r')

            final[name] = self.__average_features_dir(i, j, provider, start_date, end_date)

        final = scoring_postprocess(final)

        return final

    def get_layers(self):
        for layer in self.net.layers:
            print(layer.output)
