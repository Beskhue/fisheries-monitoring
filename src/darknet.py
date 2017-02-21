"""
Module for DarkNet interoperability.
"""

import os
import random
import math
import settings

def save_annotations_for_darknet(train_imgs, single_class = False):
    """
    Convert the bounding box annotations to the format supported by Darknet and save them.

    :param train_imgs: The training images to convert bounding boxes for.
    :param single_class: Whether to collapse fish classes to a single class (i.e., all classes become "Fish")
    """

    print("Starting annotation conversion. This may take some time to complete.")
    print("Output dir: %s" % settings.CONVERTED_BOUNDING_BOX_OUTPUT_DIR)

    for img, clss, meta in zip(train_imgs['x'], train_imgs['y'], train_imgs['meta']):
        if clss == "NoF":
            continue

        dir = os.path.join(settings.CONVERTED_BOUNDING_BOX_OUTPUT_DIR, clss)

        if not os.path.exists(dir):
            os.makedirs(dir)

        file = os.path.join(dir, meta['filename'] + ".txt")
        img = img()
        img_height, img_width, img_channels = img.shape

        with open(file, "w") as f:
            for box in meta['bounding_boxes']:
                if single_class:
                    class_index = 0
                else:
                    class_index = settings.CLASS_NAME_TO_INDEX_MAPPING[box['class']]

                abs_x = box['x']
                abs_y = box['y']
                abs_width = box['width']
                abs_height = box['height']

                x = (abs_x + abs_width / 2.0) / img_width
                y = (abs_y + abs_height / 2.0) / img_height
                width = abs_width / img_width
                height = abs_height / img_height

                f.write('%s %s %s %s %s\n' % (class_index, x, y, width, height))

    print("Annotation conversion completed.")
    print("Dumping file locations...")

    meta = [m for m in train_imgs['meta'] if m['class'] != "NoF"]
    
    random.seed(963629463)
    random.shuffle(meta)

    train_size = math.floor(len(meta) * 0.8)
    val_size = math.floor(len(meta) * 0.1)
    test_size = len(meta) - train_size - val_size

    save_image_names_to_file("train.txt", meta[0:train_size])
    save_image_names_to_file("validation.txt", meta[train_size : train_size + val_size])
    save_image_names_to_file("test.txt", meta[train_size + val_size : train_size + val_size + test_size])

    print("All done.")

def save_image_names_to_file(file_name, images):
    """
    Save image paths to a file as required by Darknet.

    :param file_name: The name of the file to write the image file names to.
    :param images: The meta information of the images for which to write the locations to the file.
    """
    if not os.path.exists(settings.CONVERTED_BOUNDING_BOX_OUTPUT_DIR):
        os.makedirs(settings.CONVERTED_BOUNDING_BOX_OUTPUT_DIR)

    file_name = os.path.join(settings.CONVERTED_BOUNDING_BOX_OUTPUT_DIR, file_name)
    with open(file_name, "w") as f:
        for image in images:
            f.write("%s/%s.jpg\n" % (image['class'], image['filename']))
