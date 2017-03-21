"""
Module to train and use neural networks.
"""

import pipeline
import settings
import keras
import scipy.misc
import numpy as np
from threadsafe import threadsafe_generator

def build():
    model = keras.models.Sequential()

    # Convolutional layers
    model.add(keras.layers.Conv2D(
            filters = 6, 
            kernel_size = (32, 32), 
            input_shape = (200, 200, 3), 
            activation = "relu"))
    model.add(keras.layers.Conv2D(
            filters = 12, 
            kernel_size = (32, 32), 
            activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Conv2D(
            filters = 18, 
            kernel_size = (16, 16), 
            activation = "relu"))
    model.add(keras.layers.Conv2D(
            filters = 24, 
            kernel_size = (16, 16), 
            activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Conv2D(
            filters = 30, 
            kernel_size = (8, 8), 
            activation = "relu"))

    # Dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=240, activation = "relu"))
    model.add(keras.layers.Dense(units=7))

    # Output layer
    model.add(keras.layers.Activation("softmax"))

    # Compile model
    model.compile(
            optimizer = 'rmsprop',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
    
    return model

def train(epochs = 100):
    """
    Train a neural net using the pipeline.
    """

    print("Building and compiling model.")
    model = build()
    model.summary()

    # Define a method to crop and resize train images
    def crop_and_resize(img, y, meta):
        if len(meta['bounding_boxes']) > 0:
            bbox = meta['bounding_boxes'][0]
            x = round(bbox['x'])
            y = round(bbox['y'])
            width = round(bbox['width'])
            height = round(bbox['height'])

            (size_y, size_x, channels) = img.shape

            #print("%s %s %s %s %s %s" % (x, y, width, height, size_x, size_y))

            if y + height < size_y and x + width < size_x and height > 0 and width > 0 and x >= 0 and y >= 0:
                img = img[y:y+height, x:x+width, :]

                #print(img.shape)
            #print("---")

        img = scipy.misc.imresize(img, size=(200, 200))

        return img

    # Create the pipeline, filtering away NoF and registering the crop and resize method
    pl = pipeline.Pipeline(class_filter = ["NoF"], f_middleware = crop_and_resize)
    class_count = pl.class_count()
    class_count_idx = {}
    for clss in class_count:
        class_count_idx[settings.CLASS_NAME_TO_INDEX_MAPPING[clss]] = float(class_count[clss]) / pl.num_unique_samples()

    generators = pl.train_and_validation_generator_generator()

    # Define a method to create a batch generator
    @threadsafe_generator
    def batch_generator(generator, batch_size = 64):
        n = 0
        for x, y, meta in generator:
            if n == 0:
                xs = []
                ys = []

            xs.append(x())
            ys.append(settings.CLASS_NAME_TO_INDEX_MAPPING[y])
            n += 1

            if n == batch_size:
                n = 0
                yield (np.array(xs), np.array(ys))

    model.fit_generator(
            batch_generator(generators['train']),
            steps_per_epoch = 10, 
            epochs = 10,
            class_weight = class_count_idx,
            validation_data = batch_generator(generators['validate']),
            validation_steps = 2,
            workers = 4)
