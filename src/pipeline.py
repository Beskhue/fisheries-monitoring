"""
Pipeline module defening the various classes required for the pipeline
"""

import os
import glob
import settings
import scipy.misc
import sklearn.model_selection

class Pipeline:
    def __init__(self):
        self.dataLoader = DataLoader()
        self.load()

    def load(self):
        """
        Load the data
        """
        self.trainData = self.dataLoader.getTrainImagesAndClasses()
        

    def trainAndValidationMiniBatchGeneratorGenerator(self, mini_batch_size = 128):
        """
        Generate train and validation mini batch generators of the training data, by
        splitting the data into train and validation sets.

        :param mini_batch_size: The size of the mini-batches
        :return: A dictionary with the training set mini-batch generator in 'train', 
                 and the validation set mini-batch generator in 'validate'
        """  
        x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(
            self.trainData['x'], 
            self.trainData['y'], 
            test_size = 0.2, 
            stratify = self.trainData['y'])

        return {
            'train': self.miniBatchGenerator(x_train, y_train, mini_batch_size = mini_batch_size),
            'validate': self.miniBatchGenerator(x_validate, y_validate, mini_batch_size = mini_batch_size)
            }

    def miniBatchGenerator(self, *x, mini_batch_size):
        """
        Generate a mini batch generator from the input arrays

        :param *x: One or more input arrays
        :param batch_size: The size of each mini-batch
        :return: A generator (iterable) of n output arrays (with n the number of input arrays), such 
                 that they are mini-batches of the input arrays.
        """
        num_args = len(x)
        if num_args == 0:
            raise TypeError("Expected at least one input")

        n = len(x[0])
        start_idx = 0

        while start_idx < n:
            last_idx = start_idx + mini_batch_size
            if last_idx > n:
                last_idx = n

            out = [[] for x in range(num_args)]

            for idx in range(start_idx, last_idx):
                for i in range(0, num_args):
                    out[i].append(x[i][idx])

            yield tuple(out)
            start_idx += mini_batch_size

class DataLoader:
    """
    Class for the various data loading routines.
    """

    def getTrainImagesAndClasses(self):
        """
        Method to load the train cases.

        :return: A dictionary containing the list of classes (y) and list of (function to load) images (x)
        """

        classes = os.listdir(settings.TRAIN_DIR)
        y = []
        x = []

        for clss in classes:
            dir = os.path.join(settings.TRAIN_DIR, clss)

            filenames = glob.glob(os.path.join(dir, "*.jpg"))
            for filename in filenames:
                x.append(lambda: self.load(filename))
                y.append(clss)


        return {'x': x, 'y': y}

    def load(self, filename):
        """
        Load an image into a scipy ndarray

        :param filename: The name of the file to load
        :return: The image as an ndarray
        """
        return scipy.misc.imread(filename)
