"""
README

Retrieves arrays representing various variables, rearranges them appropriately
and feed them into a correlation coeeficient function.

Available correlation coefficients:
- Spearman

- Directory to extract .mat files:
'C:/Users/User/Desktop/urop-neuralnetwork/proj-capsnet-keras/dissimilarity_matrices/'

"""

import numpy as np
import scipy.io as sio  # for loading 69dataset_matrix
from scipy.stats import spearmanr

class Capsnet4():

    def Spearman(self, dataset_directory69, dataset_directory1, dataset_directory2, dataset_directory3, dataset_directory1_key = 'matrix'):

        """
        README

        :param dataset_directory69: DIR of 69dataset RDM
        :param dataset_directory1: DIR of first layer RDM
        :param dataset_directory2: DIR of second layer RDM
        :param dataset_directory3: DIR of third layer RDM
        :param dataset_directory1_key: Some key
        """

        #----------------------------Symbolic constants-------------------------


        #-------------------------------------

        def dissimilarity_to_array(inputMatrix):
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

            return outputArray

        #-------------------------------------

        # Loading dissimilarity matrices and converting the upper triangles to arrays
        # print("...Capsnet - Loading dissimilarity matrices and converting the upper triangles to arrays...")
        array69 = dissimilarity_to_array(sio.loadmat(dataset_directory69)[dataset_directory1_key])  # shape (1, 4950)
        arrayHidden1 = dissimilarity_to_array(np.load(dataset_directory1))  # shape (1, 44850)
        arrayHidden2 = dissimilarity_to_array(np.load(dataset_directory2))
        arrayHidden3 = dissimilarity_to_array(np.load(dataset_directory3))

        # Concatenating necessary arrays
        # print("...Capnset - Concatenating necessary arrays...")
        matrice_69_Hidden1 = np.concatenate((array69, arrayHidden1))
        matrice_69_Hidden2 = np.concatenate((array69, arrayHidden2))
        matrice_69_Hidden3 = np.concatenate((array69, arrayHidden3))

        # Spearman correlation coefficient
        # print("...Capsnet - Spearman correlation coefficient...")
        return [spearmanr(matrice_69_Hidden1.T), spearmanr(matrice_69_Hidden2.T), spearmanr(matrice_69_Hidden3.T)]

# x = capsnet3()
# print(x.spearman())































# fin
