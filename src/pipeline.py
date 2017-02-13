"""
Pipeline module defening the various classes required for the pipeline
"""

import os
import glob
import json
import settings
import scipy.misc
import sklearn.model_selection

class Pipeline:
    def __init__(self):
        self.data_loader = DataLoader()
        self.load()

    def load(self):
        """
        Load the data
        """
        self.train_data = self.data_loader.get_train_images_and_classes()
        

    def train_and_validation_mini_batch_generator_generator(self, mini_batch_size = 128):
        """
        Generate train and validation mini batch generators of the training data, by
        splitting the data into train and validation sets.

        :param mini_batch_size: The size of the mini-batches
        :return: A dictionary with the training set mini-batch generator in 'train', 
                 and the validation set mini-batch generator in 'validate'
        """  
        x_train, x_validate, y_train, y_validate, meta_train, meta_validate = sklearn.model_selection.train_test_split(
            self.train_data['x'], 
            self.train_data['y'], 
            self.train_data['meta'],
            test_size = 0.2, 
            stratify = self.train_data['y'])

        return {
            'train': self.mini_batch_generator(x_train, y_train, meta_train, mini_batch_size = mini_batch_size),
            'validate': self.mini_batch_generator(x_validate, y_validate, meta_validate, mini_batch_size = mini_batch_size)
            }

    def mini_batch_generator(self, *x, mini_batch_size):
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

    def get_classes(self):
        """
        Get the classes present in the training set.

        :return: A list of classes present in the training set.
        """

        return os.listdir(settings.TRAIN_DIR)

    def get_bounding_boxes(self):
        """
        Get the bounding boxes of fishes in the training data.

        :return: Dictionary containing the bounding boxes.
        """
        classes = self.get_classes()

        bounding_boxes = {}

        for clss in classes:
            if clss == "NoF":
                continue

            bounding_boxes[clss] = {}

            filename = os.path.join(settings.TRAIN_BOUNDING_BOXES_DIR, clss + ".json")

            with open(filename) as data_file:
                data = json.load(data_file)
                for d in data:
                    name = self.get_file_name_part(d['filename'])
                    annotations = d['annotations']

                    bounding_boxes[clss][name] = annotations

        return bounding_boxes


    def get_train_images_and_classes(self):
        """
        Method to load the train cases.

        :return: A dictionary containing the list of classes (y) and list of (function to load) images (x)
        """

        classes = self.get_classes()
        y = []
        x = []
        m = []

        bounding_boxes = self.get_bounding_boxes()

        for clss in classes:
            dir = os.path.join(settings.TRAIN_DIR, clss)

            filenames = glob.glob(os.path.join(dir, "*.jpg"))
            for filename in filenames:
                name = self.get_file_name_part(filename)

                meta = {}
                meta['filename'] = name
                meta['class'] = clss
                if clss != "NoF":
                    meta['bounding_boxes'] = bounding_boxes[clss][name]

                x.append((lambda filename: lambda: self.load(filename))(filename))
                y.append(clss)
                m.append(meta)


        return {'x': x, 'y': y, 'meta': m}

    def load(self, filename):
        """
        Load an image into a scipy ndarray

        :param filename: The name of the file to load
        :return: The image as an ndarray
        """
        return scipy.misc.imread(filename)

    def get_file_name_part(self, full_path):
        """
        Get the name of the file (without extension) given by the specified full path.

        E.g. "/path/to/file.ext" becomes "file"

        :param full_path: The full path to the file
        :return: The name of the file without the extension (and without the path)
        """
        base = os.path.basename(full_path)
        return os.path.splitext(base)[0]
