from __future__ import print_function

import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics
#from sklearn.metrics import accuracy_score

import requests
requests.packages.urllib3.disable_warnings()
import ssl

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

from data import chexpert_data as chexdata

batch_size = 16
#nb_classes = 10
nb_epoch = 3

# XXX
img_rows, img_cols = 320, 320
img_channels = 3

# ???
img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)

depth = 19  # 121
nb_dense_block = 3  # ??
growth_rate = 12 # ??
nb_filter = -1  # ??
dropout_rate = 0.0 # 0.0 for data augmentation ??

model = densenet.DenseNet(img_dim, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, classes=14, weights=None, activation='sigmoid')
print("Model created")

model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

folder = '/Volumes/work/data/medical/CheXpert-v1.0-small'
#folder = '/home/mediratta/CheXpert-v1.0-small/'
(trainX, trainY), (testX, testY) = chexdata.load_data(folder)

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainY = trainY.astype('float32')
testY = testY.astype('float32')

trainX = densenet.preprocess_input(trainX)
testX = densenet.preprocess_input(testX)

#Y_train = np_utils.to_categorical(trainY)
#Y_test = np_utils.to_categorical(testY)
Y_train = trainY
Y_test = testY
generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0)

# Load model
weights_file="weights/DenseNet-40-12-Chexpert.h5"
if os.path.exists(weights_file):
    #model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

out_dir="weights/"

lr_reducer      = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                    cooldown=0, patience=5, min_lr=1e-5)
model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                  save_weights_only=True, verbose=1)

callbacks=[lr_reducer, model_checkpoint]

model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(testX, Y_test),
                    validation_steps=testX.shape[0] // batch_size, verbose=1)

yPreds = model.predict(testX)
#yPred = np.argmax(yPreds, axis=1)
# threshholds
yPred = list(map(lambda x: list(map(lambda y: round(y), x)), yPreds))
yPred_np = np.asarray(yPred)
yTrue = Y_test

accuracy = metrics.accuracy_score(yTrue, yPred_np) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)