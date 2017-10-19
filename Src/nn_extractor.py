
class NNExtractor:
    """
    Class
    -----
    Handles the feature extraction from a pre-trained NN.
    """
    def __init__(self, config):
        """
        Initializes the NNExtractor object where the model to be used is defined.
        :param config: the config file
        """
        self.config = config
        self.output_image_dir=self.config['output_image_dir']

        if config['model'] == 'ResNet50':
            print('INFO: loading ResNet50 ...')
            from keras.applications.resnet50 import ResNet50
            self.net = ResNet50(weights='imagenet', include_top=False)

        elif config['model'] == 'VGG16':
            print('INFO: loading VGG16 ...')
            from keras.applications.vgg16 import VGG16
            from keras.applications.vgg16 import preprocess_input
            self.net = VGG16(weights='imagenet', include_top=False)

        else:
            print("ERROR: Only ResNet50 and VGG16 implemented so far")

    def __average_features_dir(self, image_dir):
        """
        Private function that takes the average of the features computed for all the images in the cluster into one feature.
        :param image_dir: string with path to the folder with images for one tile.
        :return: a list with the averages for each feature extracted from the images in the tile.
        """
        from pandas import DataFrame
        from keras.preprocessing import image
        import os
        import numpy as np
        if self.config['model'] == 'ResNet50':
            from keras.applications.resnet50 import preprocess_input
        elif self.config['model'] == 'VGG16':
            from keras.applications.vgg16 import preprocess_input
        else:
            print('ERROR: only ResNet50 and VGG16 implemented so far')

        i = 0
        features_df = DataFrame([])

        for name in os.listdir(image_dir):
            if name.endswith(".jpg"):
                img_path = os.path.join(image_dir, name)
                img = image.load_img(img_path, target_size=(400, 400))
                image_preprocess = image.img_to_array(img)
                image_preprocess = np.expand_dims(image_preprocess, axis=0)
                image_preprocess = preprocess_input(image_preprocess)

                features = self.net.predict(image_preprocess)

                features = features.ravel()

                features_df[name] = features
        avg_features = features_df.mean(axis=1)

        return avg_features

    def extract_features(self):
        """
        Loops over the folders (tiles) and collects the features.
        :return:
        """
        from pandas import DataFrame
        import os

        Final = DataFrame([])
        for name in os.listdir(self.output_image_dir):
            if len(name) == 10:
                print('scoring images for tile (i,j): {}'.format(name))
                Final[name] = self.__average_features_dir(os.path.join(self.output_image_dir, name))

        return Final


    def get_layers(self):
        for layer in self.net.layers:
            print(layer.output)