import os
import sys
sys.path.append(os.path.join("..","Src"))
from img_lib import RasterGrid
from img_utils import getRastervalue
from sqlalchemy import create_engine
import yaml
import pandas as pd
from nn_extractor import NNExtractor
import numpy as np
import functools
import keras

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Input, Convolution2D, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

from keras.preprocessing import image
from keras import optimizers

from keras.callbacks import EarlyStopping

preprocess_input = keras.applications.resnet50.preprocess_input

dataset = "../Data/datasets/WFP_ENSAN_Senegal_2013_individual.csv"
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
data = data[['i', 'j', 'gpsLatitude', 'gpsLongitude', indicator]].groupby(["i", "j"]).mean()

print("Number of unique tiles: {} ".format(len(data)))

print(data.head())

# Transoform regression into a classification task
a = 3 #number of classes
labels = []
for i in range(0, a):
    labels.append(str(i))
data["{}_cat".format(indicator)] = pd.qcut(data[indicator], a, labels=labels)

list_i = data.index.get_level_values(0).values
list_j = data.index.get_level_values(1).values

data["j_lat"] = GRID.centroid_y_coords[list_j]
data["i_lon"] = GRID.centroid_x_coords[list_i]

for sat in provider.split(","):
    print('INFO: routine for provider: ', sat)
    # download the images from the relevant API
    GRID.download_images(list_i, list_j, step, sat, start_date, end_date)
    print('INFO: images downloaded.')


GRID.image_dir = os.path.join("../Data", "Satellite", provider + '/')


# Pre-process the images
img_rows = 350
img_cols = 350

X = []
for ind, row in data.iterrows():
    i = ind[0]
    j = ind[1]
    name = str(i) + '_' + str(j)
    lon = np.round(GRID.centroid_x_coords[i], 5)
    lat = np.round(GRID.centroid_y_coords[j], 5)
    file_name = str(lon) + '_' + str(lat) + '_' + str(16) + '.jpg'
    img_path = os.path.join(GRID.image_dir, file_name)
    if os.path.exists(img_path):
        img = image.load_img(img_path, target_size=(img_rows, img_cols))
        image_preprocess = image.img_to_array(img)
        image_resized = image_preprocess * 1. / 255
        X.append(image_resized)
    else:
        print("image {},{} not found".format(i, j))
        data.drop((i, j), inplace=True)

print(data.head())
print(data.shape)

X, data = shuffle(X, data, random_state=7)

X = np.array(X)

y = data["{}_cat".format(indicator)]
y = keras.utils.np_utils.to_categorical(y)

# split into 80% for train and 20% for test
seed = 7

# fix random seed for reproducibility
np.random.seed(7)

# Create model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(a))  # Number of classes
model.add(Activation('softmax'))

# Compile model
opt = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

from keras.preprocessing.image import ImageDataGenerator

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

train_datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

val_datagen = ImageDataGenerator()

batch_size = 10
epochs = 150
nb_train_samples = int(len(y) * 0.8)
nb_validation_samples = int(len(y) * 0.2)

steps_per_epoch = nb_train_samples // batch_size
validation_steps = nb_validation_samples // batch_size

print("steps_per_epoch: {} validation_steps: {}".format(steps_per_epoch, validation_steps))

train_iterator = train_datagen.flow(X_train, y_train)
val_iterator = val_datagen.flow(X_test, y_test)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=5,
                           verbose=0, mode='auto')
callbacks_list = [early_stop]

history = model.fit_generator(
    train_iterator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_iterator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)

# evaluate the model
score = model.evaluate(X, y, verbose=0)
print("{}: {}".format(model.metrics_names[1], score[1]))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model squared error')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_plot.png')
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('training_plot.png')


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

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
opt = optimizers.SGD(lr=0.01)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

# evaluate loaded model on test data
score = loaded_model.evaluate(X, y, verbose=0)
print("{}: {}".format(loaded_model.metrics_names[1], score[1]))

data["y"] = data["{}_cat".format(indicator)]

y_hat = loaded_model.predict(X)
print(y_hat.shape)
y_hat = np.argmax(y_hat, axis=1)
print(np.squeeze(y_hat))
data["y_hat"] = np.squeeze(y_hat)

data.to_csv("predictions.csv")
