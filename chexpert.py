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
from utils import img_util

# training variables:
batch_size = 8
nb_epoch = 3

#training data vars
training_data_sz = 223414
valid_sz = 234

def create_model(img_channels=3, img_rows=64, img_cols=64, depth=121, activation='sigmoid'):
    img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
    model = densenet.DenseNet(img_dim, depth=depth, nb_dense_block=3, growth_rate=12, nb_filter=-1, dropout_rate=0.0, classes=14, weights=None, activation=activation)
    print("Model created")
    model.summary()
    optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")
    return model


def prepare_training_data():
    folder = '/Volumes/work/data/medical/CheXpert-v1.0-small'
    #folder = '/home/mediratta/CheXpert-v1.0-small/'
    (trainX, trainY), (testX, testY) = chexdata.load_data(folder)

    _type = 'float32'
    trainX = trainX.astype(_type)
    testX = testX.astype(_type)
    trainY = trainY.astype(_type)
    testY = testY.astype(_type)

    trainX = densenet.preprocess_input(trainX)
    testX = densenet.preprocess_input(testX)

    Y_train = trainY
    Y_test = testY
    return (trainX, Y_train), (testX, Y_test)

def prepare_training_data_gen():
    folder = '/Volumes/work/data/medical/CheXpert-v1.0-small'
    #folder = '/home/mediratta/CheXpert-v1.0-small/'
    gen_train, gen_test = chexdata.load_data_gen(folder, batch_size)
    return gen_train, gen_test

def augument_training_data(trainX):
    _type = 'int8'
    trainX = trainX.astype(_type)
    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,
                                   horizontal_flip=True)
    generator = ImageDataGenerator()
#    generator.fit(trainX, seed=0)
    return generator

# Load model
def load_model(model, weights_file):
    if False and os.path.exists(weights_file):
        model.load_weights(weights_file, by_name=True)
        print("Model loaded.")
        return True, model
    else:
        return False, model

def do_training(model, generator, trainX, Y_train, testX, Y_test, weights_file):        
    lr_reducer      = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                        cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)

    callbacks=[lr_reducer, model_checkpoint]
    _var = generator.flow(trainX, Y_train, batch_size=batch_size)
    #_var = (trainX, Y_train)
    model.fit_generator(_var,
                        steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=(testX, Y_test),
                        validation_steps=testX.shape[0] // batch_size, verbose=1)
    return model

def do_training_gen(model, gen_train, gen_test, weights_file):
    lr_reducer      = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                        cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)

    callbacks=[lr_reducer, model_checkpoint]
    model.fit_generator(gen_train,
                        steps_per_epoch=training_data_sz // batch_size, epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=gen_test,
                        validation_steps=valid_sz // batch_size, verbose=1)
    return model

def do_inferencing(model, testX, base_val):
    yPreds = model.predict(testX)
    #yPred = np.argmax(yPreds, axis=1)
    # threshholds
    yPred = list(map(lambda x: list(map(lambda y: find_class(y, base_val), x)), yPreds))
    yPred_np = np.asarray(yPred)
    return yPred_np

def calculate_accuracy(model, testX, Y_test, base_val):
    yPred_np = do_inferencing(model, testX, base_val)
    accuracy = metrics.accuracy_score(Y_test, yPred_np) * 100
    error = 100 - accuracy
    print("Base_val, Accuracy, Error : ", base_val, accuracy, error)
    return accuracy, error

weights_file="weights/DenseNet-40-12-Chexpert.h5"

if __name__ == "__main__":
    model = create_model()
    gen_train, gen_test = prepare_training_data_gen()
    #generator = augument_training_data(trainX)
    _, model = load_model(model, weights_file)
    model = do_training_gen(model, gen_train, gen_test, weights_file)
    base_val = 0.0
    while base_val < 1.0:
        accuracy, error = calculate_accuracy(model, testX, Y_test, base_val)
        base_val += 0.1
        
#if __name__ == "__main__":
#    model = create_model()
#    train_gen, test_gen = get_data_generators()
##    generator = augument_training_data(trainX)
#    _, model = load_model(model, weights_file)
#    model = do_training(model, train_gen, trainX, Y_train, testX, Y_test, weights_file)
#    base_val = 0.0
#    while base_val < 1.0:
#        accuracy, error = calculate_accuracy(model, testX, Y_test, base_val)
#        base_val += 0.1        
'''
1. read image
2. format it
3. call predict
4. find class
'''
def do_inferencing_test(image_path: str, y_test):
    model = create_model()
    _, model = load_model(model, weights_file)
    image_arr = np.array(img_util.convert_image(image_path))
    y_pred = do_inferencing(model, image_arr)
    accuracy = metrics.accuracy_score(y_pred, y_test) * 100
    return accuracy

def find_class(_val: float, _base=0.5):
    if _val < _base:
        return 0
    else:
        return 1

#def generate_arrays_from_file(path):
#    while True:
#        with open(path) as f:
#            for line in f:
#                # create numpy arrays of input data
#                # and labels, from each line in the file
#                x1, x2, y = process_line(line)
#                yield ({'input_1': x1, 'input_2': x2}, {'output': y})