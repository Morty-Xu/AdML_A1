import cv2
import os
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def image_list(image_root, txt='list.txt'):
    f = open(txt, 'wt')

    for (label, filename) in enumerate(sorted(os.listdir(image_root), reverse=False)):
        if os.path.isdir(os.path.join(image_root, filename)):
            for imagename in os.listdir(os.path.join(image_root, filename)):
                name, ext = os.path.splitext(imagename)
                ext = ext[1:]
                if ext == 'jpg' or ext == 'png' or ext == 'bmp':
                    f.write('%s %d\n' % (os.path.join(image_root, filename, imagename), label))
    f.close()


def shuffle_split(list_file, trainFile, valFile):
    with open(list_file, 'r') as f:
        records = f.readlines()
    random.shuffle(records)
    num = len(records)
    trainNum = int(num * 0.8)
    with open(trainFile, 'w') as f:
        f.writelines(records[0:trainNum])
    with open(valFile, 'w') as f1:
        f1.writelines(records[trainNum:])


def mnist_dataloader(fn):

    os.makedirs('./output', exist_ok=True)

    image_root = fn + '/train_images'
    if True:
        image_list(image_root, txt='output/total.txt')
        shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')

    data_label = [k for k in range(10)]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data_label)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    train_data = []
    fh_train = open('./output/train.txt', 'r')
    for line in fh_train:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        image = np.array(cv2.imread(words[0], cv2.IMREAD_GRAYSCALE), dtype='float')
        image = np.reshape(image, (-1, 1)) / 255
        train_data.append((image, np.reshape(onehot_encoded[int(words[1])], [-1, 1])))

    val_data = []
    fh_val = open('./output/val.txt', 'r')
    for line in fh_val:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        image = np.array(cv2.imread(words[0], cv2.IMREAD_GRAYSCALE), dtype='float')
        image = np.reshape(image, (-1, 1)) / 255
        val_data.append((image, np.reshape(onehot_encoded[int(words[1])], [-1, 1])))

    return train_data, val_data







