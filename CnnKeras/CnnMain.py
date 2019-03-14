import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu, floatX=float32"

# If using tensorflow, set image dimensions order
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

import time
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, BaseLogger
import scipy.io as sio

from CnnKeras.step1 import Cnn0
from CnnKeras.step2 import Cnn1


class CnnMain():
    cnn0 = None
    cnn1 = Cnn1()

    def __init__(self, train_features, train_labels):
        # data pre-processing happens here. Each models will have separate requirements for their data
        self.cnn0 = Cnn0(train_features, train_labels)
        print(" ~~~ CnnMain initialized ~~~ ")

    def train(self, train_features, train_labels, name):
        self.cnn0.train(name)
        print(" ~~~ End of model training ~~~ ")

    def test(self, test_features, test_labels, dir):
        print(self.cnn0.test(dir, test_features, test_labels))
        print(" ~~~ End of model testing ~~~ ")

    def getActivations(self, test_features, layer_index, dir, output_dir):
        self.cnn0.activations(dir, test_features, output_dir, layer_index)
        print(" ~~~ End of Activation retrieval ~~~ ")

    def getRdm(self, activation_dir, rdm_dir):
        self.cnn1.rdm(activation_dir, rdm_dir)
        print(" ~~~ End of RDM creation ~~~ ")

    def getSpearman(self, dataset_rdm_dir, rdm_dir):
        result = self.cnn1.spear(dataset_rdm_dir, rdm_dir)
        print("Result: {0}".format(result))
        print(" ~~~ End of Spearman creation ~~~ ")


def CnnPreprocess(name, features, labels):
    # database name MNIST
    if name == "./data/MNIST/":
        # features
        features = np.squeeze(features)
        features = features.reshape(features.shape[0], 1, 28, 28).astype('float32')
        features /= 255
        # labels
        labels = np_utils.to_categorical(labels, 10)

        return features, labels



















































# fin
