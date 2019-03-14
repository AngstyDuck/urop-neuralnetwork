"""
README

Training iterated 20 times, 20 epochs and saved as trained_model_{0}.h5
Removed all functions that display img or graphs
Removed all argparsing
Testing accuracy of individual models will happen separately after all training

"""


import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from CapsnetKeras.utils import combine_images
from PIL import Image
from CapsnetKeras.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

K.set_image_data_format('channels_last')


class Capsnet1():

    # -------------------------Symbolic Constants---------------------------
    EPOCHS = 20
    BATCHSIZE = 100
    LR = 0.001  # Initial learning rate
    LRDECAY = 0.9  # The value multiplied by self.LR at each epoch. Set a larger value for larger self.EPOCHS
    LAMRECON = 0.392  # The coefficient for the loss of decoder
    ROUTINGS = 3  # Number of iterations used in routing algorithm. should > 0
    SHIFTFRACTION = 0.1  # Fraction of pixels to shift at most in each direction.
    DEBUG = False  # Save weights by TensorBoard
    SAVEDIR = "./backend/result"
    TESTING = False  # Test the trained model on self.TESTING dataset
    DIGIT = 5  # self.DIGIT to manipulate
    WEIGHTS = None  # The path of the saved weights. Should be specified when self.TESTING


    def __init__(self, feature_train, label_train, feature_test, label_test, savedir):
        self.SAVEDIR = savedir
        self.feature_train = feature_train
        self.label_train = label_train
        self.feature_test = label_test
        self.label_test = label_test

    def TrainModel(self, EPOCHS = 20, WEIGHTS = None, TESTING = False):
        self.EPOCHS = EPOCHS
        self.WEIGHTS = WEIGHTS
        self.TESTING = TESTING
        # x_train = self.feature_train
        # y_train = self.label_train
        # x_test = self.feature_test
        # y_test = self.label_test


        def CapsNet(input_shape, n_class, routings):

            """
            A Capsule Network on MNIST.
            :param input_shape: data shape, 3d, [width, height, channels]
            :param n_class: number of classes
            :param routings: number of routing iterations

            :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                    `eval_model` can also be used for training.
            """
            x = layers.Input(shape=input_shape)

            # Layer 1: Just a conventional Conv2D layer
            conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
            primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

            # Layer 3: Capsule layer. Routing algorithm works here.
            self.DIGITcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,

                                     name='self.DIGITcaps')(primarycaps[0])

            # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
            # If using tensorflow, this will not be necessary. :)
            out_caps = Length(name='capsnet')(self.DIGITcaps)

            # Decoder network.
            y = layers.Input(shape=(n_class,))
            masked_by_y = Mask()([self.DIGITcaps, y])  # The true label is used to mask the output of capsule layer. For training
            masked = Mask()(self.DIGITcaps)  # Mask using the capsule with maximal length. For prediction

            # Shared Decoder model in training and prediction
            decoder = models.Sequential(name='decoder')
            decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
            decoder.add(layers.Dense(1024, activation='relu'))
            decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
            decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

            # Models for training and evaluation (prediction)
            train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
            eval_model = models.Model(x, [out_caps, decoder(masked)])

            # manipulate model
            noise = layers.Input(shape=(n_class, 16))
            noised_digitcaps = layers.Add()([self.DIGITcaps, noise])
            masked_noised_y = Mask()([noised_digitcaps, y])
            manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
            return train_model, eval_model, manipulate_model


        def margin_loss(y_true, y_pred):
            """
            Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
            :param y_true: [None, n_classes]
            :param y_pred: [None, num_capsule]
            :return: a scalar loss value.
            """
            L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
                0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

            return K.mean(K.sum(L, 1))


        def train(model, data):
            """
            Training a CapsuleNet
            :param model: the CapsuleNet model
            :param data: a tuple containing training and self.TESTING data, like `((x_train, y_train), (x_test, y_test))
            :return: The trained model
            """
            # unpacking the data
            (x_train, y_train), (x_test, y_test) = data

            # callbacks
            log = callbacks.CSVLogger(self.SAVEDIR + '/log.csv')
            tb = callbacks.TensorBoard(log_dir=self.SAVEDIR + '/tensorboard-logs',
                                       batch_size=self.BATCHSIZE, histogram_freq=int(self.DEBUG))
            checkpoint = callbacks.ModelCheckpoint(self.SAVEDIR + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                                   save_best_only=True, save_weights_only=True, verbose=1)
            lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: self.LR * (self.LRDECAY ** epoch))

            # compile the model
            model.compile(optimizer=optimizers.Adam(lr=self.LR),
                          loss=[margin_loss, 'mse'],
                          loss_weights=[1., self.LAMRECON],
                          metrics={'capsnet': 'accuracy'})

            """
            # Training without data augmentation:
            model.fit([x_train, y_train], [y_train, x_train], batch_size=self.BATCHSIZE, epochs=self.EPOCHS
                      validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
            """

            # Begin: Training with data augmentation ---------------------------------------------------------------------#
            def train_generator(x, y, batch_size, shift_fraction=0.):
                train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                                   height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
                generator = train_datagen.flow(x, y, batch_size=batch_size)
                while 1:
                    x_batch, y_batch = generator.next()
                    yield ([x_batch, y_batch], [y_batch, x_batch])

            # Training with data augmentation. If shift_fraction=0., also no augmentation.
            model.fit_generator(generator=train_generator(x_train, y_train, self.BATCHSIZE, self.SHIFTFRACTION),
                                steps_per_epoch=int(y_train.shape[0] / self.BATCHSIZE),
                                epochs=self.EPOCHS,
                                validation_data=[[x_test, y_test], [y_test, x_test]],
                                callbacks=[log, tb, checkpoint, lr_decay])
            # End: Training with data augmentation -----------------------------------------------------------------------#

            model.save_weights(self.SAVEDIR + '/trained_model.h5')
            print('Trained model saved to \'%s/trained_model.h5\'' % self.SAVEDIR)


            return model


        def test(model, data):
            x_test, y_test = data
            y_pred, x_recon = model.predict(x_test, batch_size=100)
            print('-'*30 + 'Begin: test' + '-'*30)

            return np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]


        def manipulate_latent(model, data):
            print('-'*30 + 'Begin: manipulate' + '-'*30)
            x_test, y_test = data
            index = np.argmax(y_test, 1) == self.DIGIT
            number = np.random.randint(low=0, high=sum(index) - 1)
            x, y = x_test[index][number], y_test[index][number]
            x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
            noise = np.zeros([1, 10, 16])
            x_recons = []
            for dim in range(16):
                for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
                    tmp = np.copy(noise)
                    tmp[:,:,dim] = r
                    x_recon = model.predict([x, y, tmp])
                    x_recons.append(x_recon)

            x_recons = np.concatenate(x_recons)

            img = combine_images(x_recons, height=16)
            image = img*255
            Image.fromarray(image.astype(np.uint8)).save(self.SAVEDIR + '/manipulate-%d.png' % self.DIGIT)
            print('manipulated result saved to %s/manipulate-%d.png' % (self.SAVEDIR, self.DIGIT))
            print('-' * 30 + 'End: manipulate' + '-' * 30)


        def load_mnist():
            # the data, shuffled and split between train and test sets
            from keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
            x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
            y_train = to_categorical(y_train.astype('float32'))
            y_test = to_categorical(y_test.astype('float32'))
            return (x_train, y_train), (x_test, y_test)

        if not os.path.exists(self.SAVEDIR):
            os.makedirs(self.SAVEDIR)
        
        # load data
        # print('loading data')
        (x_train, y_train), (x_test, y_test) = load_mnist()
        print("original feature_train.shape(): {0}".format(x_train.shape))
        # print('-')
        
        # define model
        model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                      n_class=len(np.unique(np.argmax(y_train, 1))),
                                                      routings=self.ROUTINGS)
        
        # train or test
        if self.WEIGHTS is not None:  # init the model weights with provided one
            model.load_weights(self.WEIGHTS)
        if not self.TESTING:
            train(model=model, data=((x_train, y_train), (x_test, y_test)))
        else:  # as long as weights are given, will run self.TESTING
            if self.WEIGHTS is None:
                print('No weights are provided. Will test using random initialized weights.')
            manipulate_latent(manipulate_model, (x_test, y_test))
        
            return test(model=eval_model, data=(x_test, y_test))
            # return test(model=eval_model, data=(x_test, y_test))
        
        
        # Saving the entire model
        print('done')




































# end
