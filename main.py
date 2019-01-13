"""
README


"""

import os
import scipy.io as sio
import numpy as np




MODEL = "cnn"  # Available models include "cnn", (caps sensitive)
OPTIONS = "train"  # Available options include "train", (caps sensitive)

# DIR of all datasets
DATASETDIR69 = "./data/69dataset.mat"
DATASETDIRMNIST = "./data/MNIST/"


# -----------------------------------------------------------import dataset here
def ExtractData(importDataDir):
    if importDataDir == "./data/69dataset.mat":
        # load 69dataset
        testFeaturesData = sio.loadmat(importDataDir)['X']
        testFeaturesData.reshape(100,28,28)

        testLabelData = []
        for i in sio.loadmat(importDataDir)['labels']:
            element = 0
            if i == 1:
                element = 6
            elif i == 2:
                element = 9
            testLabelData.append(element)
        testLabelData = np.array(testLabelData)

        return testFeaturesData, testLabelData

    elif importDataDir == "./data/MNIST/":
        import gzip

        trainFeatures = gzip.open(importDataDir + "train-images-idx3-ubyte.gz","r")  # training set contains 60000 examples
        trainLabel = gzip.open(importDataDir + "train-labels-idx1-ubyte.gz","r")
        testFeatures = gzip.open(importDataDir + "t10k-images-idx3-ubyte.gz","r")  # testing set contains 10000 examples
        testLabel = gzip.open(importDataDir + "t10k-labels-idx1-ubyte.gz","r")

        FeaturesSize = 28
        trainNumFeatures = 60000
        testNumFeatures = 10000

        trainFeatures.read(16)
        trainFeaturesBuf = trainFeatures.read(FeaturesSize * FeaturesSize * trainNumFeatures)
        trainFeaturesData = np.frombuffer(trainFeaturesBuf, dtype=np.uint8).astype(np.float32)
        trainFeaturesData = trainFeaturesData.reshape(trainNumFeatures, FeaturesSize, FeaturesSize, 1)

        trainLabel.read(8)
        buf = trainLabel.read(1 * trainNumFeatures)
        trainLabelData = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        testFeatures.read(16)
        testFeaturesBuf = testFeatures.read(FeaturesSize * FeaturesSize * testNumFeatures)
        testFeaturesData = np.frombuffer(testFeaturesBuf, dtype=np.uint8).astype(np.float32)
        testFeaturesData = testFeaturesData.reshape(testNumFeatures, FeaturesSize, FeaturesSize, 1)

        testLabel.read(8)
        buf = testLabel.read(1 * testNumFeatures)
        testLabelData = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        return trainFeaturesData, trainLabelData, testFeaturesData, testLabelData


trainFeaturesData = ExtractData(DATASETDIRMNIST)[0]
trainLabelData = ExtractData(DATASETDIRMNIST)[1]
testFeaturesData, testLabelData = ExtractData(DATASETDIR69)
# -----------------------------------------------------------------------------

def model(option, model, dataName):
    if option == "train":
        # selecting model
        if model == "cnn":
            from CnnKeras.CnnMain import CnnMain as Model
            from CnnKeras.CnnMain import CnnPreprocess

            train_features, train_labels = CnnPreprocess(dataName, ExtractData(DATASETDIRMNIST)[0], ExtractData(DATASETDIRMNIST)[1])
            test_features, test_labels = CnnPreprocess(dataName, ExtractData(DATASETDIR69)[0], ExtractData(DATASETDIR69)[1])

            model = Model(train_features, train_labels)
            # model.train(train_features, train_labels, "boi")
            model.test(train_features, train_labels, "./CnnKeras/backend/models/69dataset/0.h5")


model("train", "cnn", DATASETDIRMNIST)
































# fin
