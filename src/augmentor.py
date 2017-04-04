import keras.preprocessing.image
import keras.backend as K
import PIL
import numpy as np
import settings

class Augmentor(keras.preprocessing.image.ImageDataGenerator):
    """
    Use Keras' Image Data Generator to augment an image
    """

    def __init__(self, imageDataGenerator, augmentation_mode):
        self.imageDataGenerator = imageDataGenerator
        self.augmentation_mode = augmentation_mode
    def random_blur(self, x):
        augmentation = settings.AUGMENTATION[self.augmentation_mode]
        radius = np.random.uniform(augmentation['BLUR_RANGE'][0], augmentation['BLUR_RANGE'][1])
        x = PIL.Image.fromarray(x.astype("uint8"))#.astype("uint8"))
        x = x.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
        x = np.array(x).astype("float32")
        return x

    def augment(self, x):
        #print(1,x.mean())
        x = self.random_blur(x)
        #print(2,x.mean())
        x = self.imageDataGenerator.random_transform(x.astype(K.floatx()))
        #print(4,x.mean())
        return x
