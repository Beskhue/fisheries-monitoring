import os
import pprint
from clize import run
import pipeline
import segmentation
import settings

def example():
    """
    Run the pipeline example (tests if the pipeline runs succesfully, should produce summary output of the first batch and first case in that batch).
    """

    pl = pipeline.Pipeline()

    a = pl.train_and_validation_mini_batch_generator_generator()
    train_mini_batches = a['train']

    x, y, meta = next(train_mini_batches)
    print("Number of cases in first batch: %s" % len(x))
    print("First image shape and label: %s - %s" % (str(x[0]().shape), y[0]))
    print("First image meta information:")
    pprint.pprint(meta[0])

def example_crop_plot():
    import scipy

    def crop_and_resize(img, y, meta):
        bbox = meta['bounding_boxes'][0]
        x = round(bbox['x'])
        y = round(bbox['y'])
        width = round(bbox['width'])
        height = round(bbox['height'])

        img_height = len(img)

        img = img[y:y+height, x:x+width, :]

        return img

    pl = pipeline.Pipeline(class_filter = ["NoF"], f_middleware = crop_and_resize)
    class_count = pl.class_count()
    class_count_idx = {}
    for clss in class_count:
        class_count_idx[settings.CLASS_NAME_TO_INDEX_MAPPING[clss]] = float(class_count[clss]) / pl.num_unique_samples()

    generators = pl.train_and_validation_generator_generator()

    (x, y, meta) = next(generators['train'])
    img = x()

    
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.ylabel('some numbers')
    plt.show()


def train_network():
    """
    Train a neural net using the pipeline.
    """
    
    import network

    network.train()
    input("Press Enter to continue...")

def train_top_xception_network():
    """
    Train the top of the extended xception network.
    """

    import network

    tl = network.TransferLearning()

    tl.build('xception', summary = False)
    tl.train_top(
        epochs = 70,
        mini_batch_size = 32)

def fine_tune_xception_network():
    """
    Fine-tune the extended xception network. To do this, first the top
    of the extended xception network must have been trained already.
    """

    import network

    tl = network.TransferLearning()

    tl.build('xception', summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        mini_batch_size = 32,
        input_weights_name = "ext_xception_toptrained.hdf5",
        n_layers = 125)

def segment_dataset(dataset, index_range=None, *, silent=False):
    """
    Segments (part of) the given data set by colour, producing and saving candidate bounding boxes in a JSON file. Note: slow!
    
    dataset: Which image data to segment: train, test or final.
    
    index_range: Which image indices to segment (either one index (4) or a range(6-7)). Warning: due to initial overhead, relatively even sloewr for one index.
    
    silent: No output.
    """
    if index_range is not None:
        try:
            index_range = [int(index_range)]
        except ValueError:
            try:
                parts = index_range.split('-',1)
                index_range = list(range(int(parts[0]), int(parts[1])+1))
            except:
                print('Not a valid index or index range: ' + index_range)
    segmentation.do_segmentation(img_idxs=index_range, output = not silent, save_candidates=True, data=dataset)

def convert_annotations_to_darknet(single_class = False):
    """
    Convert the bounding box annotations to the format supported by Darknet.

    single_class: Whether to collapse fish classes to a single class (i.e., all classes become "Fish")
    """
    
    import darknet

    dl = pipeline.DataLoader()
    train_imgs = dl.get_train_images_and_classes()
    darknet.save_annotations_for_darknet(train_imgs, single_class = single_class)

def crop_images(histogram_matching = True):
    """
    Crop images in the data using the bounding box annotations.

    Creates one crop for each bounding box.

    :param histogram_matching: Whether to apply histogram matching on the night-vision images.
    """

    import preprocessing
    from skimage.io import imsave

    print("Loading images...")

    # Load all images
    dl = pipeline.DataLoader()
    data = dl.get_train_images_and_classes()

    imgs = data['x']
    ys = data['y']
    metas = data['meta']

    print("Loaded %s images." % len(imgs))

    # Prepare for histogram matching if we need it
    if histogram_matching:
        import colour
        print("Applying histogram matching. Preparing template...")
        template = preprocessing.build_template(imgs, metas)
        print("Template prepared")


    print("Cropping %s input images to %s..." % (len(imgs), settings.CROPS_OUTPUT_DIR))

    classes_encountered = []
    n = 0

    for img, meta in zip(imgs, metas):
        # Load image
        img = img()

        if "bounding_boxes" not in meta:
            # No bounding boxes, so skip this image
            continue

        if histogram_matching:
            if colour.is_night_vision(img):
                # We are performing histogram matching and this image is night vision,
                # so histogram match it
                img = preprocessing.hist_match(img, template)

        # For each crop...
        for crop, clss in zip(*preprocessing.crop(img, meta["bounding_boxes"])):
            n += 1
            class_dir = os.path.join(settings.CROPS_OUTPUT_DIR, clss)
            file_path = os.path.join(class_dir, "img_%s.jpg" % n)

            # Create directory for class if it does not exist yet
            if clss not in classes_encountered:
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                classes_encountered.append(clss)

            # Save the crop to file
            imsave(file_path, crop)

    print("All images cropped.")


if __name__ == "__main__":
    run(example,
        example_crop_plot,
        train_network,
        train_top_xception_network,
        fine_tune_xception_network,
        segment_dataset,
        convert_annotations_to_darknet,
        crop_images)
