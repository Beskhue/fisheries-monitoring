"""
Module to train and use neural networks.
"""
import os
import abc
import functools
import pipeline
import settings
import keras
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
import scipy.misc
import numpy as np
import metrics

PRETRAINED_MODELS = {
    "vgg16":     VGG16,
    "vgg19":     VGG19,
    "inception": InceptionV3,
    "xception":  Xception,
    "resnet":    ResNet50
}

class Learning:
    def __init__(self, class_balance_method = "weights", mini_batch_size = 32, data_type = "original", class_filter = [], tensor_board = True, validate = True):
        """
        :param class_balance_method: None, "weights", or "batch"
        """

        self.model = None
        self.class_balance_method = class_balance_method
        self.mini_batch_size = mini_batch_size
        self.tensor_board = tensor_board
        self.validate = validate

        self.pl = pipeline.Pipeline(data_type = data_type, class_filter = class_filter)

        self.generators = {}
        self.generator_chain = [
            self.pl.augmented_generator,
            self.pl.drop_meta_generator,
            self.pl.class_mapper_generator,
            functools.partial(self.pl.mini_batch_generator, mini_batch_size = mini_batch_size),
            self.pl.to_numpy_arrays_generator
        ]

    def set_full_generator(self):
        """
        Add a data augmented generator over all dataset (without train
        validation splitting) to the generators dictionary
        """
        self.generators = self.pl.data_generator_builder(
            *self.generator_chain,
            infinite = True, shuffle = True,)

    def set_train_val_generators(self):
        """
        Add the train and validation data augmented generators to the 
        generators dictionary
        """
        self.generators = self.pl.train_and_validation_data_generator_builder(
            *self.generator_chain,
            balance = self.class_balance_method == "batch",
            infinite = True,
            shuffle = True)

    @abc.abstractmethod
    def build(self):
        throw(NotImplemented("Must be implemented by child class."))

    def train(self, epochs, weights_name):
        """
        Train the (previously loaded/set) model. It is assumed that the `model` object
        has been already configured (such as which layers of it are frozen and which not)
        
        :param epochs: the number of training epochs
        :param mini_batch_size: size of the mini batches
        :param weights_name: name for the h5py weights file to be written in the output folder
        """
        
        if self.validate:
            self.set_train_val_generators()
        else:
            self.set_full_generator()

        if self.class_balance_method == "weights":
            # Calculate class weights
            class_weights = self.pl.class_reciprocal_weights()

        callbacks_list = []

        # Create weight output dir if it does not exist
        if not os.path.exists(settings.WEIGHTS_OUTPUT_DIR):
            os.makedirs(settings.WEIGHTS_OUTPUT_DIR)

        # Save the model with best validation accuracy during training
        weights_name = weights_name + ".e{epoch:03d}-tloss{loss:.4f}-vloss{val_loss:.4f}.hdf5"
        weights_path = os.path.join(settings.WEIGHTS_OUTPUT_DIR, weights_name)
        checkpoint = keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor = 'val_loss',
            verbose=1,
            save_best_only = True,
            mode = 'min')
        callbacks_list.append(checkpoint)
                 
        if self.tensor_board:
            # Output tensor board logs
            tf_logs = keras.callbacks.TensorBoard(
                log_dir = settings.TENSORBOARD_LOGS_DIR,
                histogram_freq = 1,
                write_graph = True,
                write_images = True)
            callbacks_list.append(tf_logs)

        # Train
        self.model.fit_generator(
            generator = self.generators['train'],
            steps_per_epoch = int(3299/self.mini_batch_size), 
            epochs = epochs,
            validation_data = self.generators['validate'] if self.validate else None,
            validation_steps = int(0.3*3299/self.mini_batch_size) if self.validate else None,
            class_weight = class_weights if self.class_balance_method == "weights" else None,
            workers = 2,
            callbacks = callbacks_list)


class TransferLearning(Learning):
	
    def __init__(self, *args, **kwargs):
        """
        TransferLearning initialization.
        """
        super().__init__(*args, **kwargs)

        self.base_model = None
        self.base_model_name = None
        self.model_name = None

    def extend(self):
        """
        Extend the model by stacking new (dense) layers on top of the network
        """
        x = self.base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(7, activation='softmax')(x)

        # This is the model we will train:
        self.model = keras.models.Model(input=self.base_model.input, output=predictions)

    def build(self, base_model_name, input_shape = None, extended_model_name = None, summary = False):
        """
        Build an extended model. A base model is first loaded disregarding its last layers and afterwards
        some new layers are stacked on top so the resulting model would be applicable to the
        fishering-monitoring problem
        
        :param base_model_name: model name to load and use as base model (`"vgg16"`,`"vgg19"`,`"inception"`,`"xception"`,`"resnet"`).
        :param input_shape: optional shape tuple (see input_shape of argument of base network used in Keras).
        :param extended_model_name: name for the extended model. It will affect only to the weights file to write on disk
        :param summary: whether to print the summary of the extended model
        """

        # Set the base model configuration and extended model name
        self.base_model_name = base_model_name
        self.base_model = PRETRAINED_MODELS[self.base_model_name](weights = 'imagenet', include_top = False, input_shape = input_shape)
        if not extended_model_name:
            extended_model_name = 'ext_' + base_model_name

        self.model_name = extended_model_name

        # Extend the base model
        print("Building %s using %s as the base model..." % (self.model_name, self.base_model_name))
        self.extend()        
        print("Done building the model.")

        if summary:
            print(self.model.summary())
    
    def fine_tune_extended(self, epochs, input_weights_name, n_layers = 126):
        """
        Fine-tunes the extended model. It is assumed that the top part of the classifier has already been trained
        using the `train_top` method. It retrains the top part of the extended model and also some of the last layers
        of the base model with a low learning rate.
        
        :param epochs: the number of training epochs
        :param mini_batch_size: size of the mini batches
        :param input_weights_name: name of the h5py weights file to be loaded as start point (output of `train_top`).
        :param n_layers: freeze every layer from the bottom of the extended model until the nth layer. Default is
        126 which is reasonable for the Xception model
        """

        # Load weights
        self.model.load_weights(os.path.join(settings.WEIGHTS_DIR,input_weights_name))

        # Freeze layers
        for layer in self.model.layers[:n_layers]:
           layer.trainable = False

        for layer in self.model.layers[n_layers:]:
           layer.trainable = True

        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics = ['accuracy', metrics.precision, metrics.recall])
        weights_name = self.model_name+'.finetuned'
        
        # Train
        self.train(epochs, weights_name)
        
    def train_top(self, epochs):
        """
        Trains the top part of the extended model. In other words it trains the extended model but freezing every
        layer of the base model.
        
        :param epochs: training epochs
        :param mini_batch_size: size of the mini batches
        """
    
        # Freeze all convolutional base_model layers
        for layer in self.base_model.layers:
            layer.trainable = False
            
        self.model.compile(
                optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy', metrics.precision, metrics.recall])
                
        weights_name = self.model_name+'.toptrained'
        
        # Train
        self.train(epochs, weights_name)

class TransferLearningFishOrNoFish(TransferLearning):

    def extend(self):
        """
        Extend the model by stacking new (dense) layers on top of the network
        """
        x = self.base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(2, activation='softmax')(x)

        # This is the model we will train:
        self.model = keras.models.Model(input=self.base_model.input, output=predictions)


def build():
    model = keras.models.Sequential()

    # Convolutional layers    
    model.add(keras.layers.Conv2D(
            filters = 5, 
            kernel_size = (16, 16), 
            input_shape = (200, 200, 3), 
            activation = "relu"))
    model.add(keras.layers.Conv2D(
            filters = 10, 
            kernel_size = (16, 16), 
            activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Conv2D(
            filters = 20, 
            kernel_size = (16, 16), 
            activation = "relu"))
    model.add(keras.layers.Conv2D(
            filters = 30, 
            kernel_size = (16, 16), 
            activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Conv2D(
            filters = 60, 
            kernel_size = (8, 8), 
            activation = "relu"))
    model.add(keras.layers.Conv2D(
            filters = 70, 
            kernel_size = (8, 8), 
            activation = "relu"))
    model.add(keras.layers.Conv2D(
            filters = 80, 
            kernel_size = (8, 8), 
            activation = "relu"))

    # Dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=240, activation = "relu"))
    model.add(keras.layers.Dense(units=240, activation = "relu"))
    model.add(keras.layers.Dense(units=7))

    # Output layer
    model.add(keras.layers.Activation("softmax"))

    # Optimizer
    optimizer = keras.optimizers.Adagrad(lr = 0.0001)

    # Compile model
    model.compile(
            optimizer = optimizer,
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
   
    class_weights = pl.class_weights()

    generators = pl.train_and_validation_data_generator_builder(
        pl.drop_meta_generator,
        pl.class_mapper_generator,
        pl.mini_batch_generator,
        pl.to_numpy_arrays_generator)

    model.fit_generator(
            generators['train'],
            steps_per_epoch = 20, 
            epochs = 200,
            class_weight = class_weights,
            validation_data = generators['validate'],
            validation_steps = 2,
            workers = 2)


    # Evaluate network on some samples
    xs = []
    ys = []
    n = 0
    for x, y, meta in generators['validate']:
        if n >= 32:
            break
        
        n += 1

        xs.append(x())
        ys.append(settings.CLASS_NAME_TO_INDEX_MAPPING[y])

    print("Predicting some samples:")
    print("Ground truth:\n%s" % [settings.CLASS_INDEX_TO_NAME_MAPPING[y] for y in ys])
    ys_pred = model.predict_classes(np.array(xs))
    print("Predicted:\n%s" % [settings.CLASS_INDEX_TO_NAME_MAPPING[y] for y in ys_pred.tolist()])
