import tensorflow as tf

# AIM:
#given a FCS dataset, it downloads the relevant images, and fine tunes the model, then saves the weights

# import pandas as pd
# import shutil
# df = pd.read_csv('../Data/datasets/VAM_ENSA_Nigeria_national_2017_individual.csv')
#
# cnt=0
# for group, i, j in zip(df['FCS_group'],df['i'], df['j']):
#     name=str(i)+'_'+str(j)
#     cnt += 1
#     if cnt<=500:
#         print('moving imgae ', name, ' to train')
#         shutil.copy2('../Data/Satellite/Google/'+name+'.jpg', '../Data/train/'+str(group))
#     else:
#         print('moving image ', name, ' to validation')
#         shutil.copy2('../Data/Satellite/Google/' + name + '.jpg', '../Data/validate/'+str(group))



train_data_dir = '../Data/train'
validation_data_dir = '../Data/validate'
# dimensions of our images.
img_width, img_height = 400, 400

nb_train_samples = 585
nb_validation_samples = 183
epochs = 30
batch_size = 16

# build the VGG16 network
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(400,400,3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = tf.keras.models.Sequential()
top_model.add(tf.keras.layers.Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
top_model.add(tf.keras.layers.Dropout(0.5))
top_model.add(tf.keras.layers.Dense(3, activation='softmax'))

# add the model on top of the convolutional base
model = tf.keras.models.Model(inputs=base_model.input, outputs=top_model(base_model.output))


# to non-trainable (weights will not be updated)
for layer in model.layers:  # 18 leaves the last max pool out
    if ((layer.name == 'block5_conv1') | (layer.name == 'block5_conv2') | (layer.name == 'block5_conv3')): 
        layer.trainable = True
    else:
        layer.trainable = False

    print (layer.name, ' : ', layer.trainable)

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=90,
    vertical_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0,
                          write_graph=True, write_images=False)

# fine-tune the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tensorboard])

# plot training
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('training_history.png')
