import keras.preprocessing.image
import keras.backend as K


class Augmentor(keras.preprocessing.image.ImageDataGenerator):
    """
    Use Keras' Image Data Generator to augment an image
    """

    def __init__(self, augmentor):
        self.augmentor = augmentor

    def augment(self, x):
        x = self.augmentor.random_transform(x.astype(K.floatx()))
        x = self.augmentor.standardize(x)
        return x
