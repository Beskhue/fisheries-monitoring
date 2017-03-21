from clize import run
import pprint
import pipeline
import darknet
import network
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
    
    network.train()
    input("Press Enter to continue...")
    

def convert_annotations_to_darknet(single_class = False):
    """
    Convert the bounding box annotations to the format supported by Darknet.

    single_class: Whether to collapse fish classes to a single class (i.e., all classes become "Fish")
    """
    
    dl = pipeline.DataLoader()
    train_imgs = dl.get_train_images_and_classes()
    darknet.save_annotations_for_darknet(train_imgs, single_class = single_class)

if __name__ == "__main__":
    run(example, example_crop_plot, train_network, convert_annotations_to_darknet)
