import os
import yaml

class NNExtractor:
    """
    Class
    -----
    Handles the feature extraction from a pre-trained NN.
    """
    def __init__(self, output_image_dir, model_type, step):
        """
        Initializes the NNExtractor object where the model to be used is defined.
        :param config: the config file
        """
        self.model_type = model_type
        self.output_image_dir = output_image_dir
        self.step=step

        if self.model_type == 'ResNet50':
            print('INFO: loading ResNet50 ...')
            from keras.applications.resnet50 import ResNet50
            self.net = ResNet50(weights='imagenet', include_top=False, pooling=None)
            # self.base_net = ResNet50(weights='imagenet', include_top=False)
            # self.net = Model(inputs=self.base_net.input, outputs=self.base_net.get_layer('block4_pool').output)

        elif self.model_type == 'VGG16':
            print('INFO: loading VGG16 ...')
            from keras.applications.vgg16 import VGG16
            from keras.applications.vgg16 import preprocess_input
            self.net = VGG16(weights='imagenet', include_top=False, pooling='avg')

        else:
            print("ERROR: Only ResNet50 and VGG16 implemented so far")

    def load_weights(self, weights_path):
        print('INFO: loading custom weights ...')
        self.net.load_weights(weights_path)

    def __average_features_dir(self, image_dir, i,j):
        """
        Private function that takes the average of the features computed for all the images in the cluster into one feature.
        :param image_dir: string with path to the folder with images for one tile.
        :return: a list with the averages for each feature extracted from the images in the tile.
        """
        from pandas import DataFrame
        from keras.preprocessing import image
        import os
        import numpy as np
        if self.model_type == 'ResNet50':
            from keras.applications.resnet50 import preprocess_input
        elif self.model_type == 'VGG16':
            from keras.applications.vgg16 import preprocess_input
        else:
            print('ERROR: only ResNet50 and VGG16 implemented so far')

        features_df = DataFrame([])

        for a in range(-self.step, 1 + self.step):
            for b in range(-self.step, 1 + self.step):
                k = i + a
                l = j + b
                name=str(k)+'_'+str(l)

                img_path = os.path.join(image_dir, name +".jpg")

                img = image.load_img(img_path, target_size=(400, 400))  # TODO: understand target_size
                image_preprocess = image.img_to_array(img)
                image_preprocess = np.expand_dims(image_preprocess, axis=0)
                image_preprocess = preprocess_input(image_preprocess)

                features = self.net.predict(image_preprocess)
                features = features.ravel()

                features_df[name] = features
        avg_features = features_df.mean(axis=1)

        return avg_features



    def extract_features(self,list_i,list_j):
        """
        Loops over the folders (tiles) and collects the features.
        :return:
        """
        from pandas import DataFrame
        import os

        Final = DataFrame([])

        cnt=0
        for i, j in zip(list_i, list_j):
            name=str(i)+'_'+str(j)
            cnt += 1
            if cnt%10 == 0:
                print("Feature extraction : {} tiles".format(cnt))
            Final[name] = self.__average_features_dir(self.output_image_dir,i,j)

        return Final

    def get_layers(self):
        for layer in self.net.layers:
            print(layer.output)
