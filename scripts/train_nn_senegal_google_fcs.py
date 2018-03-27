import os
import sys
sys.path.append(os.path.join("..","Src"))
from master_utils import download, score_merge
from img_lib import RasterGrid
from img_utils import getRastervalue
from sqlalchemy import create_engine
import yaml
import pandas as pd
from nn_extractor import NNExtractor
import numpy as np
import functools
import keras

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
preprocess_input = keras.applications.resnet50.preprocess_input

dataset = "../Data/Datasets/WFP_ENSAN_Senegal_2013_individual.csv"
indicator = "FCS"
raster = "../Data/Geofiles/Rasters/Senegal_0005_4326_1.tif"
step = 0
provider = "Google"
start_date = None
end_date = None


data = pd.read_csv(dataset)
data = data.loc[data[indicator] > 0]
data = data.sample(frac=1, random_state=1783).reset_index(drop=True)  # shuffle data


GRID = RasterGrid(raster)
list_i, list_j = GRID.get_gridcoordinates(data)

data["i"] = list_i
data["j"] = list_j

print("Number of survey records: {} ".format(len(data)))

# Aggregate survey points at the grid level
data = data[['i', 'j', 'gpsLatitude', 'gpsLongitude', indicator]].groupby(["i", "j"]).mean().reset_index()

print("Number of unique tiles: {} ".format(len(data)))

print(data.head())

# Transoform regression into a classification task
data["{}_cat".format(indicator)] = pd.qcut(data[indicator], 3, labels=["bad", "medium", "good"])

data["j_lat"] = GRID.centroid_y_coords[data["j"]]
data["i_lon"] = GRID.centroid_x_coords[data["i"]]

list_i = data["i"]
list_j = data["j"]

for sat in provider.split(","):
    #download(id, data, GRID, list_i, list_j, raster, step, sat, start_date, end_date)
    image_dir = os.path.join("../Data", "Satellite", sat, os.path.splitext(os.path.basename(raster))[0])
    print(image_dir)

# Pre-process the images

X = []
y = []
img_rows = 128
img_cols = 128


from keras.preprocessing import image

for i, j in zip(list_i, list_j):
    name = str(i) + '_' + str(j)
    img_path = os.path.join(image_dir, name + ".jpg")
    if os.path.exists(img_path):
        img = image.load_img(img_path, target_size=(img_rows, img_cols))
        image_preprocess = image.img_to_array(img)
        #image_preprocess = np.expand_dims(image_preprocess, axis=0)
        image_preprocess = preprocess_input(image_preprocess)
        X.append(image_preprocess)
        y.append(data["{}_cat".format(indicator)].loc[(data['i'] == i) & (data['j'] == j)].cat.codes.values[0])
    else:
        print("image {},{} not found".format(i, j))


X = np.array(X)
y = keras.utils.np_utils.to_categorical(y)

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)

# split into 80% for train and 20% for test
seed = 7

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# fix random seed for reproducibility
np.random.seed(7)

# Create model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))  # Number of classes
model.add(Activation('softmax'))

# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
#
# # Fit the model
# history = model.fit(X, y, validation_split=0.2, epochs=10, batch_size=10)
#
# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.figure()
# plt.plot(history.history['categorical_accuracy'])
# plt.plot(history.history['val_categorical_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('accuracy_plot.png')
# # summarize history for loss
# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('training_plot.png')
#
#
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from keras.models import model_from_json

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(loaded_model.predict(X))
