import keras.preprocessing.image
import keras.backend as K


class Augmentor(keras.preprocessing.image.ImageDataGenerator):
    """
    Use Keras' Image Data Generator to augment an image
    """

    def __init__(self, imageDataGenerator):
        self.imageDataGenerator = imageDataGenerator

    def augment(self, x):
        x = self.imageDataGenerator.random_transform(x.astype(K.floatx()))
        x = self.imageDataGenerator.standardize(x)
        return x
