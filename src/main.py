from clize import run
import pprint
import pipeline
import darknet

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

def convert_annotations_to_darknet(single_class = False):
    """
    Convert the bounding box annotations to the format supported by Darknet.

    single_class: Whether to collapse fish classes to a single class (i.e., all classes become "Fish")
    """
    
    dl = pipeline.DataLoader()
    train_imgs = dl.get_train_images_and_classes()
    darknet.save_annotations_for_darknet(train_imgs, single_class = single_class)

if __name__ == "__main__":
    run(example, convert_annotations_to_darknet)
