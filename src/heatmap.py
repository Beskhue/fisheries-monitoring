import numpy as np
import settings
import network

class Segmenter():
    def __init__(self):
        self.network = network.LearningFullyConvolutional()
        self.network.build(weights_file = settings.HEATMAP_NETWORK_WEIGHT_NAME, num_classes = 1)

    def heatmap(self, img):
        return self.network.build_multi_scale_heatmap(img)

    def find_bounding_boxes(self, img):
        pass
