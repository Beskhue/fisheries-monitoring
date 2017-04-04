import keras
from keras.engine import Layer
import keras.backend as K

# A custom layer in Keras must implement the four following methods:
# From: https://github.com/m2dsupsdlclass/lectures-labs/blob/master/labs/04_conv_nets_2/Fully_Convolutional_Neural_Networks_rendered.ipynb
class SoftmaxMap(Layer):
    # Init function
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(SoftmaxMap, self).__init__(**kwargs)

    # There's no parameter, so we don't need this one
    def build(self,input_shape):
        pass

    # This is the layer we're interested in: 
    # very similar to the regular softmax but note the additional
    # that we accept x.shape == (batch_size, w, h, n_classes)
    # which is not the case in Keras by default.
    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    # The output shape is the same as the input shape
    def get_output_shape_for(self, input_shape):
        return input_shape
