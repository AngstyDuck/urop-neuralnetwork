"""
README
Trains and saves model, test and return accuracy of model, produce activations
of individual layers.


Adapted from: http://parneetk.github.io/blog/cnn-mnist/
"""

# Use GPU for Theano, comment to use CPU instead of GPU
# Tensorflow uses GPU by default
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


class Cnn0():

    """
    README

    is there a way to input the data and num_class from outside this code?
    """
    models = None
    OUTPUTFOLDER = './backend/chkpt-weights'
    NUMFEATURES = 100  # number of features used to create activations matrix
    EPOCHS = 20  # number of epochs to train model
    train_features = None
    train_labels = None

    def __init__(self, train_features, train_labels):

        # # load mnist dataset
        # from keras.datasets import mnist
        # train_features, train_labels = mnist.load_data()[0]
        # _, img_rows, img_cols =  train_features.shape
        # num_classes = len(np.unique(train_labels))
        # num_input_nodes = img_rows*img_cols
        #
        # # load 69dataset
        # test_features = sio.loadmat(MAT69)['X']
        # test_features.reshape(100,28,28)
        #
        # test_labels = []
        # for i in sio.loadmat(MAT69)['labels']:
        #     element = 0
        #     if i == 1:
        #         element = 6
        #     elif i == 2:
        #         element = 9
        #     test_labels.append(element)
        # test_labels = np.array(test_labels)
        #
        # # data preprocessing
        # train_features = train_features.reshape(train_features.shape[0], 1, img_rows, img_cols).astype('float32')
        # test_features = test_features.reshape(test_features.shape[0], 1, img_rows, img_cols).astype('float32')
        # train_features /= 255
        # test_features /= 255
        # # convert class labels to binary class labels
        # train_labels = np_utils.to_categorical(train_labels, num_classes)
        # test_labels = np_utils.to_categorical(test_labels, num_classes)

        self.train_features = train_features
        self.train_labels = train_labels

        num_classes = train_labels.shape[1]
        np.random.seed(2017)

        # model
        self.model = Sequential()
        self.model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), padding='same'))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), input_shape=(1, 28, 28), padding='same'))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(num_classes))
        self.model.add(Activation("softmax"))

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def accuracy(self, test_x, test_y, model):
        result = self.model.predict(test_x)
        predicted_class = np.argmax(result, axis=1)
        true_class = np.argmax(test_y, axis=1)
        num_correct = np.sum(predicted_class == true_class)
        accuracy = float(num_correct)/result.shape[0]
        return (accuracy * 100)

    def train(self,inp1):
        """
        :param inp1: name of trained model;
        """
        # creating checkpoint
        if not os.path.exists(self.OUTPUTFOLDER):
            os.makedirs(self.OUTPUTFOLDER)
        filepath=self.OUTPUTFOLDER+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
                                     save_best_only=False, save_weights_only=True, \
                                     mode='auto', period=1)
        callbacks_list = [checkpoint]

        # Train the model
        model_info = self.model.fit(self.train_features, self.train_labels, batch_size=128, \
                                 epochs=self.EPOCHS, callbacks=callbacks_list, verbose=0, validation_split=0.2)

        # saving trained model
        self.model.save_weights(inp1)

    def test(self,inp1,inp2,inp3):
        """
        :param inp1: name of trained model
        :param inp2: test feature
        :param inp3: test labels
        """
        self.model.load_weights(inp1)  # loading model weights

        return self.accuracy(inp2, inp3, self.model)

    def activations(self,inp1,inp2,inp3,inp4):
        """
        :param inp1: name of trained model
        :param inp2: test feature
        :param inp3: name of activation matrix (for saving)
        :param inp4: index of layer to retrieve activations; DIR of images
        """
        # loading model weights
        self.model.load_weights(inp1)

        # loading activations from individual layer
        get_layer_output = K.function([self.model.layers[0].input],
                              [self.model.layers[inp4].output])
        layer_output = get_layer_output([inp2[:self.NUMFEATURES]])[0]
        print(layer_output.shape)
        np.save(inp3,layer_output)


















































































# fin
