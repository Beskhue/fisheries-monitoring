import keras.preprocessing.image
import keras.backend as K
import PIL
import numpy as np
import settings

class Augmentor(keras.preprocessing.image.ImageDataGenerator):
    """
    Use Keras' Image Data Generator to augment an image
    """

    def __init__(self, imageDataGenerator):
        self.imageDataGenerator = imageDataGenerator

    def random_blur(self, x):
        radius = np.random.uniform(settings.AUGMENTATION_BLUR_RANGE[0], settings.AUGMENTATION_BLUR_RANGE[1])
        x = PIL.Image.fromarray(x.astype("uint8"))
        x = x.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
        x = np.array(x).astype("float32")
        return x

    def augment(self, x):
        # x = self.random_blur(x)
        x = self.imageDataGenerator.random_transform(x.astype(K.floatx()))
        return x
