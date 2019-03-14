import numpy as np
from scipy.stats import spearmanr


class Cnn1():
    def dissimilarity_matrix(self, inputArray):
        """
        Takes array of [index of activations, flattened pixels of activations], and
        produce dissimilarity matrix of (activation1 - activation2)**2.

        Args
        - array: Numpy array of array of [index of activations, flattened pixels of
        activations]

        Returns
        - dissimilarity matrix as numpy array of shape (index of activations, index
        of activations)

        Requires
        - module numpy as np
        """
        arrayShape = inputArray.shape

        # create empty output array
        outputMatrix = np.empty([arrayShape[0],arrayShape[0]])

        # fill up output array
        for j in range(arrayShape[0]):
            for i in range(arrayShape[0]):
                outputMatrix[j,i] = np.sum((inputArray[i,:] - inputArray[j,:])**2)

        return outputMatrix

    def dissimilarity_to_array(self, inputMatrix):
        """
        Takes a dissimilarity matrix (where matrix is symmetric along the diagonal
        from upper left corner) and returns an array created from the upper triangle

        Args
        Dissimilarty matrix as numpy

        Returns
        Single dimensional array as numpy of shape (1, __)

        """
        matrixShape = inputMatrix.shape

        # Create empty array same shape as output
        subHorizontalIndex = 0
        outputColumnLength = sum(range(matrixShape[0]))
        outputArray = np.empty((1 ,outputColumnLength))

        # change elements of the empty array accordingly
        horizontalInputIndex = 0
        inputLeftIndex = 0
        outputLeftIndex = 0
        outputRightIndex = 0
        for i in range(matrixShape[0]):
            inputLeftIndex += 1
            outputLeftIndex = outputRightIndex
            outputRightIndex += matrixShape[0] - inputLeftIndex
            outputArray[:,outputLeftIndex:outputRightIndex] = inputMatrix[i,inputLeftIndex:]

        print('output array shape: {0}'.format(outputArray.shape))
        return outputArray


    # ----------------------------------------------------------------------------

    def rdm(self, inp1, inp2):
        """
        :param inp1: name of activation matrix;
        :param inp2: name of rdm
        """
        activation = np.load(inp1)  # load activations
        rdmMatrix = self.dissimilarity_matrix(activation)  # create rdm
        np.save(inp2,rdmMatrix)  # Save rdm

    def spear(self, inp1, inp2):
        """
        :param inp1: name of 69dataset rdm;
        :param inp2: name of rdm;
        """
        array69 = self.dissimilarity_to_array(np.load(inp1))
        array = self.dissimilarity_to_array(np.load(inp2))  # shape (1, 44850)
        print('spearmanr: {0}'.format(spearmanr(array69[0], array[0])))
        return spearmanr(array69[0], array[0])


































# fin
