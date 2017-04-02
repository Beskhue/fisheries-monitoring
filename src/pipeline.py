"""
Pipeline module defening the various classes required for the pipeline
"""

import os
import glob
import json
import collections
import itertools
import functools
import settings
import scipy.misc
import sklearn.model_selection
import numpy as np
import threading
from time import sleep
from threadsafe import threadsafe_generator

class Pipeline:

    def __init__(self, data_type = "original", dataset = "train", class_filter = [], f_middleware = lambda *x: x[0]):
        """
        Pipeline initialization.

        :param data_type: The source data the pipeline should use ("original", "ground_truth_cropped", "candidates_cropped")
        :param class_filter: A list of classes to ignore (i.e., they won't be loaded)
        :param f_middleware: A function to execute on the loaded raw image, its class and the meta-inform
        """
        self.f_middleware = f_middleware
        self.data_loader = DataLoader(class_filter)

        self.class_to_index_mapper = lambda clss: settings.CLASS_NAME_TO_INDEX_MAPPING[clss]

        if data_type == "original":
            self.load_original(dataset = dataset)
        elif data_type == "ground_truth_cropped":
            self.load_precropped_ground_truth()
        elif data_type == "candidates_cropped":
            self.load_precropped_candidates(dataset = dataset)
            self.class_to_index_mapper = lambda clss: 0 if clss == "negative" else 1
        else:
            throw(ValueError("data_type should be 'original' or 'ground_truth_cropped'. Got: %s" % data_type))

    def load_original(self, dataset):
        """
        Load the data
        """
        self.data = self.data_loader.get_original_images(dataset = dataset, f_middleware = self.f_middleware)
        
    def load_precropped_ground_truth(self):
        """
        Load the pre-cropped data based on the ground-truth bounding boxes
        """
        self.data = self.data_loader.get_precropped_ground_truth_train_images(self.f_middleware)

    def load_precropped_candidates(self, dataset):
        """
        Load the pre-cropped data based on the candidates
        """
        self.data = self.data_loader.get_precropped_candidates_images(dataset = dataset, f_middleware = self.f_middleware)
    
    def get_data(self):
        return self.data

    def _data_generator(self, xs, ys, metas, infinite = False, shuffle = False):
        """
        A generator to yield (loaded) images, targets and meta information.

        :param xs: List of functions to load images
        :param ys: List of targets
        :param metas: List of meta information
        :param infinite: Whether the generator should loop infinitely over the data
        :param shuffle: Whether the data should be shuffled (and reshuffled each loop of the generator)

        :return: A generator yielding a tuple of (loaded) images, targets and meta information.
        """
        for n in itertools.count():
            if shuffle:
                xs, ys, metas = sklearn.utils.shuffle(xs, ys, metas, random_state = n)

            for x, y, meta in zip(xs, ys, metas):
                yield x(), y, meta

            if not infinite:
                break

    @threadsafe_generator
    def _threadsafe_generator(self, generator):
        for g in generator:
            yield g

    def _chain_generators(self, generator, *generators):
        """
        Chain "modifier" generators to the output of another generator.

        :param generator: The initial (deepest) generator.
        :param generators: Zero or more generators to chain to the generator.

        :return: The resulting generator yielding output as modified throughout the chain.
        """
        for generator_ in generators:
            generator = generator_(generator)
        return self._threadsafe_generator(generator)

    def data_generator_builder(self, *generators, infinite = False, shuffle = False):
        """
        Build a data generator.

        :param infinite: Whether the generator should loop infinitely over the data.
        :param shuffle: Whether the data should be shuffled (and reshuffled each loop of the generator).
        :param generators: The "modifier" generators to apply.

        :return: The generator.
        """
        g = self._data_generator(self.data['x'], self.data['y'], self.data['meta'], infinite = infinite, shuffle = shuffle)
        return self._chain_generators(g, *generators)

    def train_and_validation_data_generator_builder(self, *generators, balance = False, infinite = False, shuffle = False):
        """
        Build a data generator.

        :param balance: Whether to balance the classes.
        :param infinite: Whether the generator should loop infinitely over the data.
        :param shuffle: Whether the data should be shuffled (and reshuffled each loop of the generator).
        :param generators: The "modifier" generators to apply.

        :return: A dictionary with the training set generator in 'train', 
                 and the validation set generator in 'validate'
        """
        x_train, x_validate, y_train, y_validate, meta_train, meta_validate = sklearn.model_selection.train_test_split(
            self.data['x'], 
            self.data['y'], 
            self.data['meta'],
            test_size = 0.2,
            random_state = 7,
            stratify = self.data['y'])

        # Partially apply the _data_generator method, such that we have already applied infinite and shuffle
        data_gen = functools.partial(self._data_generator, infinite = infinite, shuffle = shuffle)

        if balance:
            # We want to balance the data. First separate the data of each class:

            class_to_train_data = collections.defaultdict(list)
            class_to_validate_data = collections.defaultdict(list)

            for x, y, meta in zip(x_train, y_train, meta_train):
                class_to_train_data[y].append((x, y, meta))

            for x, y, meta in zip(x_validate, y_validate, meta_validate):
                class_to_validate_data[y].append((x, y, meta))

            def balanced_generator_build(class_to_data):
                # Create a list of infinite generators for the data in each seperate class
                generators = []

                for clss in class_to_data:
                    x, y, meta = zip(*class_to_data[clss])
                    generators.append(data_gen(x, y, meta))

                # Continuously yield from the various generators in a balanced way
                try:
                    while 1:
                        for generator in generators:
                                yield next(generator)
                except GeneratorExit:
                    # One of the generator is done (only happens if the generators aren't infinite)
                    pass

            return {
                'train': self._chain_generators(balanced_generator_build(class_to_train_data), *generators),
                'validate': self._chain_generators(balanced_generator_build(class_to_validate_data), *generators)
                }

        else:
            # We don't want to balance the classes

            return {
                'train': self._chain_generators(data_gen(x_train, y_train, meta_train), *generators),
                'validate': self._chain_generators(data_gen(x_validate, y_validate, meta_validate), *generators),
                }

    def augmented_generator(self, generator):
        """
        Generates an augmented generator with data from an input generator
        
        :param generator: data to feed the generator

        :return: A generator augmenting the data coming from an input generator
        """ 

        import augmentor
        from keras.preprocessing.image import ImageDataGenerator

        imagegen = ImageDataGenerator(
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
        
        augm = augmentor.Augmentor(imagegen) 

        for x, y, meta in generator:
            yield augm.augment(x), y, meta

    def drop_meta_generator(self, generator):
        for x, y, meta in generator:
            yield x, y

    def image_resize_generator(self, generator, size = (300, 300)):
        for g in generator:
            g = list(g)
            g[0] = scipy.misc.imresize(g[0], size)
            yield tuple(g)

    def mini_batch_generator(self, generator, as_numpy_array = True, mini_batch_size = 32):
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
        output = []

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

    def class_mapper_generator(self, generator):
        for x, y in generator:
            yield x, self.class_to_index_mapper(y)

    def to_numpy_arrays_generator(self, generator):
        for x, y in generator:
            yield np.array(x), np.array(y)

    def num_unique_samples(self):
        """
        Count the number of unique samples in the training (and validation) data

        :return: Number of unique samples
        """
        return len(self.data['y'])

    def class_count(self):
        """
        Count the number of occurrences of each class.

        :return: Dictionary of class counts
        """

        class_count = {}
        for y in self.data['y']:
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
            class_weight[self.class_to_index_mapper(clss)] = float(num_samples) / class_count[clss] / num_classes

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
            class_weight[self.class_to_index_mapper(clss)] = float(m) / class_count[clss] * factor

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

        return os.listdir(settings.TRAIN_ORIGINAL_IMAGES_DIR)

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

            filename = os.path.join(settings.TRAIN_GROUND_TRUTH_BOUNDING_BOXES_DIR, clss + ".json")

            with open(filename) as data_file:
                data = json.load(data_file)
                for d in data:
                    name = self.get_file_name_part(d['filename'])
                    annotations = d['annotations']

                    bounding_boxes[clss][name] = annotations

        return bounding_boxes
    
    def get_candidates(self, dataset='train'):
        """
        Get the candidates of fishes in the given data set.

        :return: Dictionary containing the bounding boxes.
        """
        classes = self.get_classes()
        candidates = {}
        if dataset == 'train':
            for clss in classes:
                candidates[clss] = {}
            cand_dir = settings.TRAIN_CANDIDATES_BOUNDING_BOXES_DIR
        elif dataset == 'test':
            cand_dir = settings.TEST_CANDIDATES_BOUNDING_BOXES_DIR
        elif dataset == 'final':
            print('Final data set candidates not generated yet')
            exit()
        else:
            print('Unknown candidate data set: ' + dataset)
            exit()

        for cand_file_name in glob.glob(os.path.join(cand_dir, '*.json')):
            with open(cand_file_name) as data_file:
                data = json.load(data_file)
                for d in data:
                    name = d['filename']
                    annotations = d['candidates']
                    
                    if dataset == 'train':
                        for clss in classes:
                            if clss + "_candidates" in cand_file_name:
                                for annotation in annotations:
                                    annotation['class'] = clss
                                candidates[clss][name] = annotations
                    else:
                        candidates[name] = annotations
        
        return candidates
    
    def get_precropped_ground_truth_train_images(self, f_middleware = lambda *x: x[0], file_filter = None):
        """
        Method to load the pre-cropped ground truth train cases.

        :param f_middleware: A function to execute on the loaded raw image, its class and the meta-information
                             right after loading it. Should return the (pre-processed) image.
        :param file_filter: A list of file names (in the form of 'img_01234') to limit the output to.

        :return: A dictionary containing the list of classes (y) and list of (function to load) images (x), as well
                 as a list of meta information for each image (meta).
        """

        classes = self.get_classes()
        y = []
        x = []
        m = []

        with open(os.path.join(settings.TRAIN_GROUND_TRUTH_CROPPED_IMAGES_DIR, "_keys.json"), 'r') as infile:
            keys = json.load(infile)

        for clss in classes:
            if clss in self.class_filter:
                continue

            dir = os.path.join(settings.TRAIN_GROUND_TRUTH_CROPPED_IMAGES_DIR, clss)

            filenames = glob.glob(os.path.join(dir, "*.jpg"))
            for filename in filenames:
                name = self.get_file_name_part(filename)
                
                if file_filter is not None and name not in file_filter:
                    continue

                meta = {}
                meta['filename'] = name
                meta['class'] = clss
                meta['original_image'] = keys[name]

                x.append((lambda filename, clss, meta: lambda: f_middleware(self.load(filename), clss, meta))(filename, clss, meta))
                y.append(clss)
                m.append(meta)


        return {'x': x, 'y': y, 'meta': m}

    def get_precropped_candidates_images(self, dataset = "train", f_middleware = lambda *x: x[0], file_filter = None):
        """
        Method to load the pre-cropped candidates train cases.

        :param dataset: The dataset to get data for (train, test)
        :param f_middleware: A function to execute on the loaded raw image, its class and the meta-information
                             right after loading it. Should return the (pre-processed) image.
        :param file_filter: A list of file names (in the form of 'img_01234') to limit the output to.

        :return: A dictionary containing the list of classes (y) and list of (function to load) images (x), as well
                 as a list of meta information for each image (meta).
        """

        y = []
        x = []
        m = []

        if dataset == "train":
            classes = ["positive", "negative"]
            with open(os.path.join(settings.TRAIN_CANDIDATES_CROPPED_IMAGES_DIR, "_keys.json"), 'r') as infile:
                keys = json.load(infile)
        elif dataset == "test":
            classes = [None]
            with open(os.path.join(settings.TEST_CANDIDATES_CROPPED_IMAGES_DIR, "_keys.json"), 'r') as infile:
                keys = json.load(infile)

        for clss in classes:
            if clss in self.class_filter:
                continue

            if dataset == "train":
                dir = os.path.join(settings.TRAIN_CANDIDATES_CROPPED_IMAGES_DIR, clss)
            elif dataset == "test":
                dir = os.path.join(settings.TEST_CANDIDATES_CROPPED_IMAGES_DIR)

            filenames = glob.glob(os.path.join(dir, "*.jpg"))
            for filename in filenames:
                name = self.get_file_name_part(filename)
                
                if file_filter is not None and name not in file_filter:
                    continue

                meta = {}
                meta['filename'] = name
                meta['class'] = clss
                meta['original_image'] = keys[name]

                x.append((lambda filename, clss, meta: lambda: f_middleware(self.load(filename), clss, meta))(filename, clss, meta))
                y.append(clss)
                m.append(meta)


        return {'x': x, 'y': y, 'meta': m}
    
    def get_original_images(self, dataset = "train", f_middleware = lambda *x: x[0], file_filter = None):
        """
        Method to load the original train cases.

        :param dataset: The dataset to get data for (train, test)
        :param f_middleware: A function to execute on the loaded raw image, its class and the meta-information
                             right after loading it. Should return the (pre-processed) image.
        :param file_filter: A list of file names (in the form of 'img_01234') to limit the output to.
        :return: A dictionary containing the list of classes (y) and list of (function to load) images (x), as well
                 as a list of meta information for each image (meta).
        """

        classes = self.get_classes()
        y = []
        x = []
        m = []

        if dataset == "train":
            bounding_boxes = self.get_bounding_boxes()
        
        candidates = self.get_candidates(dataset = dataset)

        if dataset == "train":
            for clss in classes:
                if clss in self.class_filter:
                    continue

                dir = os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, clss)

                filenames = glob.glob(os.path.join(dir, "*.jpg"))
                for filename in filenames:
                    name = self.get_file_name_part(filename)
                
                    if file_filter is not None and name not in file_filter:
                        continue

                    meta = {}
                    meta['filename'] = name
                    meta['class'] = clss
                    if clss != "NoF":
                        meta['bounding_boxes'] = bounding_boxes[clss][name]
                    if name in candidates[clss]:
                        meta['candidates'] = candidates[clss][name]

                    x.append((lambda filename, clss, meta: lambda: f_middleware(self.load(filename), clss, meta))(filename, clss, meta))
                    y.append(clss)
                    m.append(meta)

            return {'x': x, 'y': y, 'meta': m}

        elif dataset == "test":
            for filename in glob.glob(os.path.join(settings.TEST_ORIGINAL_IMAGES_DIR, '*.jpg')):
                name = self.get_file_name_part(filename)
            
                if file_filter is not None and name not in file_filter:
                    continue
            
                meta = {}
                meta['filename'] = name
                if name in candidates:
                    meta['candidates'] = candidates[name]
            
                x.append((lambda filename, meta: lambda: f_middleware(self.load(filename), meta))(filename, meta))
                m.append(meta)
        
            return {'x': x, 'meta': m}

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
