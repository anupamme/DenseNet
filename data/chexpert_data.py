import cv2
import numpy as np
import os

from utils import csv_reader as csv

WIDTH = 320
HEIGHT = 320
CHANNEL = 3
'''
returns (trainX, trainY), (testX, testY)
trainX.shape: 50000, 320, 320, 3
trainY.shape: 50000, 14

testX.shape: 10000, 320, 320, 3
testY.shape: 10000, 14

may have to change the resolution as images may be in some other resolution.

image_folder:
    train.csv
    train/
    valid.csv
    valid/

read train.csv:
    ignore 1st line
    global_nd_array = []
    for each line:
        1st part is image path
        last part is label
        image_nd_array = convert_image(image_path)
        global_nd_array.append()
'''

'''
replace:
    empty by 0
    -1 by 0
'''
def replace_label(item):
    if item == '':
        return 0.0
    else:
        _fitem = float(item)
        if _fitem == -1.0:
            return 0.0
        else:
            return _fitem
            

def load_data_sub(_file):
    x_data = []
    x_label = []
    csv_data = csv.read_csv(_file)
    csv_to_use = csv_data[1:]
    base_path = '/Volumes/work/data/medical'
#    base_path = '/home/mediratta/'
    for idx, parts in enumerate(csv_to_use):
        if idx == 100:
            break
        rel_path = parts[0]
        label_vec = list(map(lambda x: replace_label(x), parts[5:]))
        image = convert_image(os.path.join(base_path, rel_path))
        x_data.append(image)
        x_label.append(label_vec)
    return x_data, x_label

def load_data(image_folder):
    train_file = os.path.join(image_folder, 'train.csv')
    x_train, label_train = load_data_sub(train_file)
    valid_file = os.path.join(image_folder, 'valid.csv')
    x_valid, label_valid = load_data_sub(valid_file)
    return (np.array(x_train), np.array(label_train)), (np.array(x_valid), np.array(label_valid))

def convert_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA)
    return img