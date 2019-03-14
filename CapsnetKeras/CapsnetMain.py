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

from CapsnetKeras.capsnet1_trainModel import Capsnet1  # function: TrainModel
from CapsnetKeras.capsnet2_produceActivations import Capsnet2  # function: ProduceActivations
from CapsnetKeras.capsnet3_dissimilarityMatrix import Capsnet3  # function: DissimilarityMatrix
from CapsnetKeras.capsnet4_spearman import Capsnet4  # function: Spearman

class CapsnetMain():
	feature_train = None
	label_train = None
	feature_test = None
	label_test = None
	savedir = None

	def __init__(self, feature_train, label_train, feature_test, label_test, savedir):
		self.feature_train = feature_train
		self.label_train = label_train
		self.feature_test = feature_test
		self.label_test = label_test
		self.savedir = savedir

	def train(self):
		capsnet1 = Capsnet1(self.feature_train, self.label_train, self.feature_test, self.label_test, self.savedir)
		capsnet1.TrainModel()

def CapsnetPreprocess(name, features, labels):
	if name == "./data/MNIST/":
		return features, labels
