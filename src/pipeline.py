"""
Pipeline module defening the various classes required for the pipeline
"""

import os
import glob
import json
import collections
import itertools
import settings
import scipy.misc
import sklearn.model_selection
import numpy as np
from time import sleep
from keras.preprocessing.image import ImageDataGenerator

class Pipeline:

    def __init__(self, class_filter = [], f_middleware = lambda x: x):
        """
        Pipeline initialization.

        :param class_filter: A list of classes to ignore (i.e., they won't be loaded)
        :param f_middleware: A function to execute on the loaded raw image, its class and the meta-inform
        """
        self.f_middleware = f_middleware
        self.data_loader = DataLoader(class_filter)
        self.load()

    def load(self):
        """
        Load the data
        """
        self.train_data = self.data_loader.get_train_images_and_classes(self.f_middleware)
        
    def load_precropped(self):
        """
        Load the pre-cropped data
        """
        self.precropped_train_data = self.data_loader.get_precropped_train_images_and_classes()
    
    def augmented_generator_generator(self, x, y, m, mini_batch_size):
        """
        Generates an augmented generator
        
        :param x: data to feed the generator
        :param y: labels of the data
        :mini_batch_size size of the mini-batches

        :return: An generator implementing data augmentation
        """ 
        datagen = ImageDataGenerator(
                rescale = settings.AUGMENTATION_RESCALE,
                rotation_range = settings.AUGMENTATION_ROTATION_RANGE,
                shear_range = settings.AUGMENTATION_SHEAR_RANGE,
                zoom_range = settings.AUGMENTATION_ZOOM_RANGE,
                width_shift_range = settings.AUGMENTATION_WIDTH_SHIFT_RANGE,
                height_shift_range = settings.AUGMENTATION_HEIGHT_SHIFT_RANGE,
                horizontal_flip = settings.AUGMENTATION_HORIZONTAL_FLIP,
                vertical_flip = settings.AUGMENTATION_VERTICAL_FLIP,
                channel_shift_range = settings.AUGMENTATION_CHANNEL_SHIFT_RANGE
                )        
        
        augmented_generator = datagen.flow(x, y, mini_batch_size)
        #TODO integrate meta information to the flow
        
        return augmented_generator
        
    def augmented_full_generator_generator(self, mini_batch_size = 128):
        """
        Use data augmentation to generate a generator for ALL the dataset.
        :mini_batch_size size of the mini-batches

        :return: A dictionary with the full set generator in 'full'
        """  
        self.load_precropped()

        x = self.precropped_train_data['x']
        y = self.precropped_train_data['y']
        meta = self.precropped_train_data['meta']
        
        return {
            'full': self.augmented_generator_generator(x, y, meta, mini_batch_size)
            }
                
    def augmented_train_and_validation_generator_generator(self, mini_batch_size = 128):
        """
        Use data augmentation to generate train and validation generators of the training data, by
        splitting the data into train and validation sets.
        :mini_batch_size size of the mini-batches

        :return: A dictionary with the training set generator in 'train', 
                 and the validation set generator in 'validate'
        """  
        self.load_precropped()
        x_train, x_validate, y_train, y_validate, meta_train, meta_validate = sklearn.model_selection.train_test_split(
            self.precropped_train_data['x'], 
            self.precropped_train_data['y'], 
            self.precropped_train_data['meta'],
            test_size = 0.2, 
            stratify = self.precropped_train_data['y'],
            random_state = 7)
        with open(os.path.join(settings.DATA_DIR,'validation_precropped_meta_set.txt'),'w') as fp:
            fp.write(str(meta_validate))
        return {
            'train': self.augmented_generator_generator(x_train, y_train, meta_train, mini_batch_size),
            'validate': self.augmented_generator_generator(x_validate, y_validate, meta_validate, mini_batch_size)
            }
    
    def train_and_validation_generator_generator(self):
        """
        Generate train and validation generators of the training data, by
        splitting the data into train and validation sets.

        :return: A dictionary with the training set generator in 'train', 
                 and the validation set generator in 'validate'
        """  
        x_train, x_validate, y_train, y_validate, meta_train, meta_validate = sklearn.model_selection.train_test_split(
            self.train_data['x'], 
            self.train_data['y'], 
            self.train_data['meta'],
            test_size = 0.2, 
            stratify = self.train_data['y'],
            random_state = 7)
        with open(os.path.join(settings.DATA_DIR,'validation_meta_set.txt'),'w') as fp:
            fp.write(str(meta_validate))
        def train_generator():
            while 1:
                for x, y, meta in zip(x_train, y_train, meta_train):
                    yield (x, y, meta)

        def validate_generator():
            while 1:
                for x, y, meta in zip(x_validate, y_validate, meta_validate):
                    yield (x, y, meta)

        return {
            'train': train_generator(),
            'validate': validate_generator()
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

    def num_unique_samples(self):
        """
        Count the number of unique samples in the training (and validation) data

        :return: Number of unique samples
        """
        return len(self.train_data['y'])
        
    def train_and_validation_generator_generator(self, balance = False):
        """
        Generate infinite train and validation generators of the training data, by
        splitting the data into train and validation sets.

        :param balance: Boolean indicating whether classes should be balanced in the output.
        :return: A dictionary with the training set generator in 'train', 
                 and the validation set generator in 'validate'
        """

        x_train, x_validate, y_train, y_validate, meta_train, meta_validate = sklearn.model_selection.train_test_split(
            self.train_data['x'], 
            self.train_data['y'], 
            self.train_data['meta'],
            test_size = 0.2, 
            stratify = self.train_data['y'])

        if balance:
            # We want to balance the data. First separate the data of each class:

            class_to_train_data = collections.defaultdict(list)
            class_to_validate_data = collections.defaultdict(list)

            for x, y, meta in zip(x_train, y_train, meta_train):
                class_to_train_data[y].append((x, y, meta))

            for x, y, meta in zip(x_validate, y_validate, meta_validate):
                class_to_validate_data[y].append((x, y, meta))

            def train_generator():
                # Create a list of infinite generators for the data in each seperate class
                generators = []

                for clss in class_to_train_data:
                    generators.append(itertools.cycle(class_to_train_data[clss]))

                while 1:
                    for generator in generators:
                        yield next(generator)

            def validate_generator():
                # Create a list of infinite generators for the data in each seperate class
                generators = []

                for clss in class_to_validate_data:
                    generators.append(itertools.cycle(class_to_validate_data[clss]))

                while 1:
                    for generator in generators:
                        yield next(generator)

            return {
                'train': train_generator(),
                'validate': validate_generator()
                }

        else:
            # We don't want to balance the classes

            def train_generator():
                while 1:
                    for x, y, meta in zip(x_train, y_train, meta_train):
                        yield (x, y, meta)

            def validate_generator():
                while 1:
                    for x, y, meta in zip(x_validate, y_validate, meta_validate):
                        yield (x, y, meta)

            return {
                'train': train_generator(),
                'validate': validate_generator()
                }


    def train_and_validation_mini_batch_generator_generator(self, mini_batch_size = 128, balance = False):
        """
        Generate (infinite) train and validation mini batch generators of the training data, by
        splitting the data into train and validation sets.

        :param mini_batch_size: The size of the mini-batches.
        :param balance: Boolean indicating whether classes should be balanced in the output.
        :return: A dictionary with the training set mini-batch generator in 'train', 
                 and the validation set mini-batch generator in 'validate'.
        """  

        generators = self.train_and_validation_generator_generator(balance = balance)

        return {
            'train': self.mini_batch_generator(generators['train'], mini_batch_size = mini_batch_size),
            'validate': self.mini_batch_generator(generators['validate'], mini_batch_size = mini_batch_size)
            }

    def mini_batch_generator(self, generator, mini_batch_size):
        """
        Generate a mini batch generator from an input generator.

        :param generator: A (potentially infinite) generator (generating tuples (including singletons))
                          to create mini batches for.
        :param batch_size: The size of each mini-batch.
        :return: A generator (iterable) of such that it generates arrays of the
                 requested size, using the output from the input generator. Each
                 mini batch has the same number of arrays as the number of elements
                 in the tuple generated by the input generator.
        """

        n = 0

        for g in generator:
            if n == 0:
                output = []

            output.append(g)
            n += 1

            if n >= mini_batch_size:
                n = 0
                yield zip(*output)

        if len(output) > 0:
            yield zip(*output)


    def class_count(self):
        """
        Count the number of occurrences of each class.

        :return: Dictionary of class counts
        """

        class_count = {}
        for y in self.train_data['y']:
            if y not in class_count:
                class_count[y] = 0

            class_count[y] += 1

        return class_count

    def class_weights(self):
        """
        Compute the class weight of each class such that the average weight
        (taking into account the frequency of each class) is 1.

        For example, if all classes have the same number of observations,
        they are all weighted 1. If we have three classes where class 1 has 
        150 observations, class 2 has 40 observations, and class 3 has 10 
        observations, they get the following weights:

        Class 1: (200/150)/3 = 0.444
        Class 2: (200/50)/3 = 1.333
        Class 3: (200/10)/3  = 6.667

        :return: The class weight of each class.
        """

        class_count = self.class_count()
        class_weight = {}
        num_samples = self.num_unique_samples()
        num_classes = len(class_count)
        
        for clss in class_count:
            class_weight[settings.CLASS_NAME_TO_INDEX_MAPPING[clss]] = float(num_samples) / class_count[clss] / num_classes

        return class_weight

    def class_reciprocal_weights(self, factor = None):
        """
        Compute the class weight of each class based on the number of occurrences
        of the most common class.

        The weight is calculated such that the classes get a weight of:
            (count of most-common class) / (class count)
        So the most-common class gets a weight of 1, and other classes get a weight
        higher than 1 (before multiplying by the factor).

        :param factor: The factor with which the multiply the weights. If not set, 
                       it is 1 / (number of classes).
        :return: The class weight based on the number of occurrences of the most
                 common class.
        """

        class_count = self.class_count()
        class_weight = {}
        m = max(class_count.values())

        if factor == None:
            factor = 1.0 / len(class_count)

        for clss in class_count:
            class_weight[settings.CLASS_NAME_TO_INDEX_MAPPING[clss]] = float(m) / class_count[clss] * factor

        return class_weight

class DataLoader:
    """
    Class for the various data loading routines.
    """

    def __init__(self, class_filter = []):
        """
        :param class_filter: A list of classes to ignore (i.e., they won't be loaded)
        """

        self.class_filter = class_filter

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
    
    def get_precropped_train_images_and_classes(self):
        """
        Method to load all the pre-cropped train cases into memory.
        
        :return: An array with all the images (X) with shape (n_images,299,299,3) and an array with all the
                 labels (Y) with shape (n_images,) 
        """

        classes = self.get_classes()
        y = []
        x = []
        m = []
        
        #print('Loading pre-cropped images from ',settings.CROPPED_TRAIN_DIR,'...')
        for clss in classes:
            filenames = glob.glob(os.path.join(settings.CROPPED_TRAIN_DIR,clss,'*'))
            
            #print("Loading ",len(fpaths),clss)
            for filename in filenames:
                name = self.get_file_name_part(filename)

                meta = {}
                meta['filename'] = name
                meta['class'] = clss
                m.append(meta)
                im = self.load(filename)[:299,:299,:]
                x.append(im.reshape(-1,299,299,3))
                y.append(settings.CLASS_NAME_TO_INDEX_MAPPING[clss])
        
        x = np.vstack(x)
        y = np.hstack(y)
        
        return {'x':x, 'y':y, 'meta': m}
    
    
    def get_train_images_and_classes(self, f_middleware):
        """
        Method to load the train cases.

        :param f_middleware: A function to execute on the loaded raw image, its class and the meta-information
                             right after loading it. Should return the (pre-processed) image.
        :return: A dictionary containing the list of classes (y) and list of (function to load) images (x), as well
                 as a list of meta information for each image (meta).
        """

        classes = self.get_classes()
        y = []
        x = []
        m = []

        bounding_boxes = self.get_bounding_boxes()

        for clss in classes:
            if clss in self.class_filter:
                continue

            dir = os.path.join(settings.TRAIN_DIR, clss)

            filenames = glob.glob(os.path.join(dir, "*.jpg"))
            for filename in filenames:
                name = self.get_file_name_part(filename)

                meta = {}
                meta['filename'] = name
                meta['class'] = clss
                if clss != "NoF":
                    meta['bounding_boxes'] = bounding_boxes[clss][name]

                x.append((lambda filename, clss, meta: lambda: f_middleware(self.load(filename), clss, meta))(filename, clss, meta))
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
