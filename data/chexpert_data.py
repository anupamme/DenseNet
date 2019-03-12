import numpy as np
import os

from utils import csv_reader as csv
from utils import img_util as img

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
            return 1   # multi_class classification
        else:
            return _fitem

def process_line(parts):
    base_path = '/Volumes/work/data/medical'
#    base_path = '/home/mediratta/'
    rel_path = parts[0]
    label_vec = list(map(lambda x: replace_label(x), parts[5:]))
    image = img.convert_image(os.path.join(base_path, rel_path))
    return image, label_vec

def load_data_sub(_file):
    x_data = []
    x_label = []
    csv_data = csv.read_csv(_file)
    csv_to_use = csv_data[1:]
    for idx, parts in enumerate(csv_to_use):
        if idx == 10000:
            break
        image, label_vec = process_line(parts)
        x_data.append(image)
        x_label.append(label_vec)
    return x_data, x_label

def load_data(image_folder):
    train_file = os.path.join(image_folder, 'train.csv')
    x_train, label_train = load_data_sub(train_file)
    valid_file = os.path.join(image_folder, 'valid.csv')
    x_valid, label_valid = load_data_sub(valid_file)
    return (np.array(x_train), np.array(label_train)), (np.array(x_valid), np.array(label_valid))

def generate_batch_size(path:str, batch_size: int):
    csv_data = csv.read_csv(path)
    csv_to_use = csv_data[1:]
    features = []
    target = []
    for idx, parts in enumerate(csv_to_use):
        image, label_vec = process_line(parts)
        features.append(image)
        target.append(label_vec)
        if (idx + 1) % batch_size == 0:
            yield (np.array(features), np.array(target))
            features = []
            target = []
    yield (np.array(features), np.array(target))