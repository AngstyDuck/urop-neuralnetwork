"""
README

Train CapsNet referenced from XiFeng Guo and save dissimilarity matrices
as .npy files for all 3 available layers.

- NOTE GLOBAL VARIABLES: conv1, primarycaps, digitcaps
- 3 hidden layers: conv1, primarycaps, digitcaps
- Using model.layers, output of these layers are:
--- conv1 = model.layers[1].output
--- primarycaps = model.layers[4].output
--- digitcaps = model.layers[5].output

"""

# for capsnet
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from CapsnetKeras.utils import combine_images
from PIL import Image
from CapsnetKeras.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import os

# additional modules for creaing RDM
import math
import scipy.io as sio



class Capsnet2():

    # -------------------------Symbolic Constants---------------------------
    EPOCHS = 20
    BATCHSIZE = 100
    LR = 0.001  # Initial learning rate
    LRDECAY = 0.9  # The value multiplied by self.LR at each epoch. Set a larger value for larger self.EPOCHS
    LAMRECON = 0.392  # The coefficient for the loss of decoder
    ROUTINGS = 3  # Number of iterations used in routing algorithm. should > 0
    SHIFTFRACTION = 0.1  # Fraction of pixels to shift at most in each direction.
    DEBUG = False  # Save weights by TensorBoard
    SAVEDIR = "./result"
    TESTING = False  # Test the trained model on self.TESTING dataset
    DIGIT = 5  # self.DIGIT to manipulate
    WEIGHTS = None  # The path of the saved weights. Should be specified when self.TESTING

    def ProduceActivations(self,
                           trainModelDir='./result/trained_model.h5',
                           saveDirList = ["./layer_activations/capsnet_layer1_activation.npy",
                                          "./layer_activations/capsnet_layer2_activation.npy",
                                          "./layer_activations/capsnet_layer3_activation.npy"]):

        # -----------------------------Symbolic constants------------------------------
        inputDataDir = './69data.npy'
        inputLabelsDir = './69labels.npy'

        key_name = 'matrix'  # key for the dictionary of activations that is saved in .mat file
        layerList = ['conv1', 'primarycaps', 'digitcaps']  # name of all hidden layers
        saveDirectoryList = saveDirList  # to save activations

        train_model_dir = trainModelDir  # load pretrained weights
        num_class = 10  # To ensure the model works, we program it to anticipate 10 different digits althogh 69data only has 2 unique digits


        def one_hot_convert(inputArray):
            """
            Creates arrayUniqueLabels that contains the unique types of labels
            available in inputArray. Then iterates through elements in inputArray,
            assigning the value of one to a one-hot array based on the index of the
            elements within arrayUniqueLabels.

            Args
            - inputArray: numpy array of shape (x,1) representing the label of the
            datasets where x represents the number of data in the dataset

            Returns
            one-hot numpy array of shape (x,y) where y represents the number of type of
            labels in the dataset

            Requires
            None
            """
            numberElements = inputArray.shape[0]
            arrayUniqueLabels = np.unique(inputArray)
            # numberUniqueLabels = len(arrayUniqueLabels)
            numberUniqueLabels = num_class
            outputArray = np.zeros((numberElements,numberUniqueLabels))

            for i in range(numberElements):
                inputElement = int(inputArray[i])
                index = int(np.where(arrayUniqueLabels==inputElement)[0][0])
                outputArray[i,index] = 1

            return outputArray

        def CapsNet(input_shape, n_class, routings):
            """
            A Capsule Network on MNIST.
            :param input_shape: data shape, 3d, [width, height, channels]
            :param n_class: number of classes
            :param routings: number of routing iterations
            :param kwargs: for usage in variable test_masked to output activations of
                hidden layers. key is 'layer', value is a string of either 'conv1',
                'primarycaps', or 'digitcaps'
            :return: Two Keras Models, the first one used for training, and the second
                one for evaluation. `eval_model` can also be used for training.
            """
            x = layers.Input(shape=input_shape)

            # Layer 1: Just a conventional Conv2D layer
            conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
            primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

            # Layer 3: Capsule layer. Routing algorithm works here.
            digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                     name='digitcaps')(primarycaps[0])

            # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
            # If using tensorflow, this will not be necessary. :)
            out_caps = Length(name='capsnet')(digitcaps)

            # Decoder network.
            y = layers.Input(shape=(n_class,))
            masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training

            # Shared Decoder model in training and prediction
            decoder = models.Sequential(name='decoder')
            decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
            decoder.add(layers.Dense(1024, activation='relu'))
            decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
            decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

            # Models for training and evaluation (prediction)
            train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])

            return train_model

        def activation_hidden_layer(model, data):
            x_test, y_test = data
            return model.predict(x_test, batch_size=100)

        # print(args)
        if not os.path.exists(self.SAVEDIR):
            os.makedirs(self.SAVEDIR)

        # load 69data and 69labels
        x_test = np.load(inputDataDir)
        x_test = np.expand_dims(x_test, axis=len(x_test.shape))
        y_test = one_hot_convert(np.load(inputLabelsDir))

        # define model
        model = CapsNet(input_shape=x_test.shape[1:],
                                                      n_class=num_class,
                                                      routings=self.ROUTINGS)

        # load weights of trained model
        model.load_weights(train_model_dir)

        # load activations of individual layers
        actList = [K.function([model.layers[0].input],
                              [model.layers[1].output]),
                   K.function([model.layers[0].input],
                              [model.layers[4].output]),
                   K.function([model.layers[0].input],
                              [model.layers[5].output])]

        # saving activations of hidden layers as dict with 'matrix' key as .mat file
        for i in range(len(actList)):
            np.save(saveDirectoryList[i], actList[i]([x_test])[0])



































# end
