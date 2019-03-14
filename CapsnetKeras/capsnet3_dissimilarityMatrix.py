"""
README
Creates dissimilarity matrix using hidden layer activations in
./layer_activations/, produces dissimilarity matrices and saves as .npy file.

"""

import numpy as np




# -----------------------------------------------------------------------------

class Capsnet3():

    def DissimilarityMatrix(self, inputDIRList, outputDIRList):

        """
        README

        :param inputDIRList: List of DIR for activation of individual layers
        :param outputDIRList: List of DIR for RDMs of each layer
        """

        # ------------------------Symbolic constants---------------------------


        def dissimilarity_matrix(inputArray):
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


        # ----------------------------------------------------------------------------
        inputList = inputDIRList
        outputList = outputDIRList
        if len(inputList)==len(outputList):
            for i in range(len(inputList)):
                # load activations
                activation = np.load(inputList[i])

                # create rdm
                # print('Creating rdm')
                rdm = dissimilarity_matrix(activation)

                # Save rdm
                np.save(outputList[i], rdm)
                # print('{0} out of {1} files done...'.format(i+1, len(inputList)))
        else:
            print('len(inputList) != len(outputList)')

        # print('capsnet3 done.')


# x = capsnet2()
# x.DissimilarityMatrix()


























# fin
