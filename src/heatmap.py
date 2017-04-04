import skimage.feature
import skimage.morphology
import skimage.measure
import numpy as np
import settings
import network

class Segmenter():
    def __init__(self):
        self.network = network.LearningFullyConvolutional()
        self.network.build(weights_file = settings.HEATMAP_NETWORK_WEIGHT_NAME, num_classes = 1)

    def heatmap(self, img):
        return self.network.build_multi_scale_heatmap(img)

    def normalize(self, heatmap):
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) 

    def find_high_activation(self, heatmap):
        # return np.maximum(heatmap - np.percentile(heatmap, q = 0.75), 0)
        return (heatmap > 0.85) * 1

    def find_bounding_boxes(self, img):

        def display_img_and_heatmap(img, heatmap, bounding_boxes):
            import matplotlib.pyplot as plt
            import matplotlib.patches
            import scipy.misc

            plt.figure(figsize=(12, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(img.astype("uint8"))
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, interpolation='nearest', cmap="viridis")
            plt.axis('off')
            ax = plt.subplot(1, 3, 3)
        
            plt.imshow(img.astype("uint8"))
            heatmap_resized = scipy.misc.imresize(heatmap, img.shape)
            plt.imshow(heatmap_resized, interpolation='nearest', cmap="viridis", alpha=0.5)
            #for blob in blobs:
            #    y, x, r = blob
            #    plt.Circle((x, y), r, UnicodeTranslateError="red", linewidth=1, fill=False)

            for bounding_box in bounding_boxes:
                rect = matplotlib.patches.Rectangle((bounding_box['x'], bounding_box['y']), bounding_box['width'], bounding_box['height'], linewidth = 1, edgecolor = 'r', fill = False)
                ax.add_patch(rect)


            plt.axis('off')
            plt.show()

        heatmap = self.heatmap(img)

        # Get a binary image of regions with high activation
        heatmap = self.find_high_activation(heatmap)
        
        # Morphologically open and then close the image to remove islands and remove gaps
        heatmap = skimage.morphology.binary_opening(heatmap)
        heatmap = skimage.morphology.binary_closing(heatmap)


        # Find connected regions in the binary image
        labeled_heatmap = skimage.measure.label(heatmap, connectivity = 1)
        region_properties = skimage.measure.regionprops(labeled_heatmap)

        (img_width, img_height, img_channels)  = img.shape
        (heatmap_width, heatmap_height) = heatmap.shape

        # Generate bounding boxes
        bounding_boxes = []

        for region_prop in region_properties:
            if region_prop['area'] <= 10:
                continue

            y1, x1, y2, x2 = region_prop['bbox']
            w = x2 - x1
            h = y2 - y1

            box = {
                'x': round(float(x1) / heatmap_width * img_width),
                'y': round(float(y1) / heatmap_height * img_height),
                'width': round(float(w) / heatmap_width * img_width),
                'height': round(float(h) / heatmap_height * img_height)
            }

            bounding_boxes.append(box)

        print(bounding_boxes)

        # blobs_doh = skimage.feature.blob_doh(heatmap)

        display_img_and_heatmap(img, heatmap, bounding_boxes)
