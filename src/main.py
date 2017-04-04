import os
import pprint
import json
from clize import run, parameters
import numpy as np
import pipeline
import settings

def example():
    """
    Run the pipeline example (tests if the pipeline runs succesfully, should produce summary output of the first batch and first case in that batch).
    """

    pl = pipeline.Pipeline(data_type = "original")

    generator = pl.data_generator_builder(pl.mini_batch_generator)

    x, y, meta = next(generator)
    print("Number of cases in first batch: %s" % len(x))
    print("First image shape and label: %s - %s" % (str(x[0].shape), y[0]))
    print("First image meta information:")
    pprint.pprint(meta[0])

def example_train_and_validation_split():
    """
    Run the pipeline example (tests if the pipeline runs succesfully, should produce summary output of the first batch and first case in that batch).
    """

    pl = pipeline.Pipeline(data_type = "ground_truth_cropped")

    generator = pl.train_and_validation_data_generator_builder(pl.mini_batch_generator, balance = True, infinite = True)

    x, y, meta = next(generator['train'])
    print("Number of cases in first batch: %s" % len(x))
    print("First image shape and label: %s - %s" % (str(x[0].shape), y[0]))
    print("First image meta information:")
    pprint.pprint(meta[0])

    print("Class counts:")
    class_counts = {}
    for clss in y:
        if clss not in class_counts:
            class_counts[clss] = 0

        class_counts[clss] += 1

    pprint.pprint(class_counts)

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

    generators = pl.train_and_validation_data_generator_builder()

    (x, y, meta) = next(generators['train'])
    img = x

    
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.ylabel('some numbers')
    plt.show()

def example_augmentation():

    import matplotlib.pyplot as plt

    pl = pipeline.Pipeline(data_type = "ground_truth_cropped")

    generator = pl.data_generator_builder(pl.augmented_generator)
    
    for x, y, meta in generator:
        
        plt.figure()
        plt.imshow(x.astype("uint8"))
        plt.axis("off")
        plt.show()

def example_fully_convolutional():

    import heatmap as hm

    # Load the heatmap segmenter
    segmenter = hm.Segmenter()

    pl = pipeline.Pipeline(data_type = "original")
    data = pl.get_data()
    
    i = 0
    for x, y, meta in zip(data['x'], data['y'], data['meta']):

        i += 1

        if y == "ALB":
            continue

        print("Index: %s" % (i - 1))
        x = x()

        # Find the bounding boxes
        segmenter.find_bounding_boxes(x, display = True)

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

    tl = network.TransferLearning(data_type = "candidates_cropped", class_balance_method = "batch", class_filter = ["NoF"])

    tl.build('xception', input_shape = (300,300,3), summary = False)
    tl.train_top(epochs = 70)

def fine_tune_xception_network():
    """
    Fine-tune the extended xception network. To do this, first the top
    of the extended xception network must have been trained already.
    """

    import network

    tl = network.TransferLearning(data_type = "candidates_cropped", class_balance_method = "batch", class_filter = ["NoF"])

    tl.build('xception', input_shape = (300,300,3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "ext_xception_toptrained.hdf5",
        n_layers = 125)

def train_top_resnet_network():
    """
    Train the top of the extended resnet50 fish type classification network.
    """

    import network

    tl = network.TransferLearning(data_type = "ground_truth_cropped", class_balance_method = "batch", class_filter = ["NoF"])

    tl.build('resnet', input_shape = (300,300,3), summary = True)
    tl.train_top(epochs = 70)

def fine_tune_resnet_network():
    """
    Fine-tune the extended resnet50 fish or no fish network. To do this, first the top
    of the extended resnet50 network must have been trained already.
    """

    import network

    tl = network.TransferLearning(data_type = "ground_truth_cropped", class_balance_method = "batch", class_filter = ["NoF"])

    tl.build('resnet', input_shape = (300,300,3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "ext_resnet_toptrained.hdf5",
        n_layers = 125)

def train_top_vgg_network():
    """
    Train the top of the extended vgg19 network.
    """

    import network

    tl = network.TransferLearning(data_type = "candidates_cropped", class_balance_method = "batch", class_filter = ["NoF"])

    tl.build('vgg19', input_shape = (300,300,3), summary = False)
    tl.train_top(epochs = 70)

def fine_tune_vgg_network():
    """
    Fine-tune the extended vgg19 network. To do this, first the top
    of the extended vgg19 network must have been trained already.
    """

    import network

    tl = network.TransferLearning(data_type = "candidates_cropped", class_balance_method = "batch", class_filter = ["NoF"])

    tl.build('vgg19', input_shape = (300,300,3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "ext_xception_toptrained.hdf5",
        n_layers = 17)

def train_top_localizer_vgg16_network():

    import network

    tl = network.TransferLearningLocalization(data_type = "original", class_filter = ["NoF"])

    tl.build('vgg16', input_shape = (256, 256, 3), summary = True)
    tl.train_top(epochs = 70)

def fine_tune_localizer_vgg16_network():
    
    import network

    tl = network.TransferLearningLocalization(data_type = "original", class_filter = ["NoF"])

    tl.build('vgg16', input_shape = (256, 256, 3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "localizer.ext_vgg16.toptrained.e007-tloss0.0198-vloss0.0417.hdf5",
        n_layers = 0)

def train_top_fish_or_no_fish_network():
    """
    Train the top of the extended xception fish or no fish network.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_candidates_cropped")

    tl.build('xception', summary = False)
    tl.train_top(epochs = 70)

def fine_tune_fish_or_no_fish_network():
    """
    Fine-tune the extended xception fish or no fish network. To do this, first the top
    of the extended xception network must have been trained already.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_candidates_cropped")

    tl.build('xception', summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "fishnofish.ext_xception.toptrained.e001-tloss0.3366-vloss0.2445.hdf5",
        n_layers = 125)

def train_top_fish_or_no_fish_resnet_network():
    """
    Train the top of the extended xception fish or no fish network.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_candidates_cropped")

    tl.build('resnet', input_shape = (300, 300, 3), summary = False)
    tl.train_top(epochs = 100)

def fine_tune_fish_or_no_fish_resnet_network():
    """
    Fine-tune the extended xception fish or no fish network. To do this, first the top
    of the extended xception network must have been trained already.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_candidates_cropped")

    tl.build('resnet', input_shape = (300, 300, 3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "fishnofish.ext_resnet.toptrained.e034-tloss0.2205-vloss0.2006.hdf5",
        n_layers = 25)

def train_top_ground_truth_fish_or_no_fish_resnet_network():
    """
    Train the top of the extended resnet fish or no fish network.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_ground_truth_cropped")

    tl.build('resnet', input_shape = (300, 300, 3), summary = False)
    tl.train_top(epochs = 100)

def fine_tune_ground_truth_fish_or_no_fish_resnet_network():
    """
    Fine-tune the extended resnet fish or no fish network. To do this, first the top
    of the extended resnet network must have been trained already.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_ground_truth_cropped")

    tl.build('resnet', input_shape = (300, 300, 3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "fishnofish.ground_truth.ext_resnet.toptrained.e041-tloss0.1008-vloss0.0756.hdf5",
        n_layers = 75)

def train_top_fish_or_no_fish_vgg_network():
    """
    Train the top of the extended inceptionv4 fish or no fish network.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_candidates_cropped")
    
    tl.build('vgg19', input_shape = (300, 300, 3), summary = False)
    
    tl.train_top(epochs = 70)

def fine_tune_fish_or_no_fish_vgg_network():
    """
    Fine-tune the extended inceptionv4 fish or no fish network. To do this, first the top
    of the extended inceptionv4 network must have been trained already.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "fish_no_fish_candidates_cropped")

    tl.build('vgg19', input_shape = (300, 300, 3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "ext_vgg19.toptrained.e060-tloss0.5310-vloss0.5097.hdf5",
        n_layers = 17)

def segment_dataset(dataset, index_range=None, *, type="colour", silent=False):
    """
    Segments (part of) the given data set by colour, producing and saving candidate bounding boxes in a JSON file. Note: slow!
    
    dataset: Which image data to segment: train, test or final.
    type: Whether to segment based on colour or a fully convolutional network

    index_range: Which image indices to segment (either one index (4) or a range(6-7)). Warning: due to initial overhead, relatively even slower for one index.
    
    silent: No output.
    """

    import segmentation

    if index_range is not None:
        try:
            index_range = [int(index_range)]
        except ValueError:
            try:
                parts = index_range.split('-',1)
                index_range = list(range(int(parts[0]), int(parts[1])+1))
            except:
                print('Not a valid index or index range: ' + index_range)

    if type == "colour":
        segmentation.do_segmentation(img_idxs=index_range, output = not silent, save_candidates=True, data=dataset)
    elif type == "fullyconv":

        import heatmap as hm

        segmenter = hm.Segmenter()

        pl = pipeline.Pipeline(data_type = "original", dataset = dataset)
        data = pl.get_data()

        if index_range == None:
            img_idxs = list(range(len(data['x'])))
        else:
            img_idxs = index_range

        outdir = settings.SEGMENTATION_CANDIDATES_OUTPUT_DIR
        os.makedirs(outdir)

        if dataset == "train":
            bounding_boxes = {}
            for clss in pl.get_classes():
                bounding_boxes[clss] = {}
                bounding_boxes[clss]['filename'] = '%s_candidates%s.json' % (clss, ('_%d-%d' % (min(img_idxs), max(img_idxs))))
                bounding_boxes[clss]['bounding_boxes'] = []
        else:
            filename = '%s_candidates%s.json' % (clss, ('_%d-%d' % (min(img_idxs), max(img_idxs))))
            bounding_boxes = []

        for idx in img_idxs:
            x = data['x'][idx]
            y = data['y'][idx]
            meta = data['meta'][idx]
            
            x = x()
            bboxes = segmenter.find_bounding_boxes(x)

            if dataset == "train":
                bounding_boxes[y]['bounding_boxes'].append({'filename': meta['filename'], 'candidates': bboxes})
            else:
                bounding_boxes.append({'filename': meta['filename'], 'candidates': bboxes})

        if dataset == "train":
            for clss in pl.get_classes():
                filename = bounding_boxes[clss]['filename']
                with open(os.path.join(outdir, filename), 'w') as outfile:
                    json.dump(bounding_boxes[clss]['bounding_boxes'], outfile)
        else:
            with open(os.path.join(outdir, filename), 'w') as outfile:
                json.dump(bounding_boxes, outfile)


def convert_annotations_to_darknet(single_class = False):
    """
    Convert the bounding box annotations to the format supported by Darknet.

    single_class: Whether to collapse fish classes to a single class (i.e., all classes become "Fish")
    """
    
    import darknet

    dl = pipeline.DataLoader()
    train_imgs = dl.get_original_images()
    darknet.save_annotations_for_darknet(train_imgs, single_class = single_class)

def crop_images(dataset, *, 
                crop_type : parameters.one_of(
                    ("candidates", "Create crops using the candidate regions."),
                    ("fullyconv", "Create crops using the candidate regions generated by the fully convolutional network."),
                    ("ground_truth", "Create crops using the ground truth fish bounding boxes."), case_sensitive = True) = "candidates", 
                num_FPs = 5,
                no_histogram_matching = False):
    """
    Crop images in the data using either bounding box annotations or generated candidates. Creates one crop for each bounding box / candidate.
    
    dataset: which data set to crop images from (train, test, final)

    crop_type: whether to use candidate regions or ground-truth bounding boxes for cropping
    
    num_FPs: if the ground truth is used, defines the number of false positives (NoF crops) per image
    
    no_histogram_matching: disable histogram matching to colour in night-vision images
    """

    import preprocessing
    import random
    from skimage.io import imsave
    
    random.seed(42) # random NoF crop reproducability
    ground_truth = crop_type == "ground_truth"

    if ground_truth and dataset != 'train':
        print("No ground truth available for dataset " + dataset)
        exit()

    print("Loading images...")

    # Load all images
    dl = pipeline.DataLoader()
    
    if dataset == 'train':
        data = dl.get_original_images(dataset = "train")
    elif dataset == 'test':
        data = dl.get_original_images(dataset = "test")
    elif dataset == 'final':
        print("Final data set support pending")
        exit()
    else:
        print("Unknown data set: " + dataset)
        exit()

    imgs = data['x']
    metas = data['meta']

    print("Loaded %s images." % len(imgs))

    # Prepare for histogram matching if we need it
    if not no_histogram_matching:
        import colour
        print("Applying histogram matching. Preparing template...")
        hist_template_data_imgs = dl.get_original_images(file_filter=preprocessing.DEFAULT_HIST_MATCH_TEMPLATES)
        template = preprocessing.build_template(hist_template_data_imgs['x'], hist_template_data_imgs['meta'])
        print("Template prepared")
    
    if ground_truth:
        metastr = "bounding_boxes"
        zoom_factor = 0.7
    else:
        if crop_type == "candidates":
            zoom_factor = 0.7
            metastr = "candidates"
        elif crop_type == "fullyconv":
            zoom_factor = 1
            metastr = "candidates_fullyconv"
        
        pos_overlap_ratio = 0.65
        pos_minor_overlap_ratio = 0.1
        pos_containment_ratio = 0.50
        pos_minor_containment_ratio = 0.17
        neg_overlap_ratio = 0.10
        
        intersection = lambda cand, fish: max(0, min(cand['x']+cand['width'], fish['x']+fish['width']) - max(cand['x'], fish['x'])) * max(0, min(cand['y']+cand['height'], fish['y']+fish['height']) - max(cand['y'], fish['y']))
        containment_ratio = lambda cand, fish: intersection(cand, fish) / float(fish['width']*fish['height'])

        contains_most_of_fish = lambda cand, fish: containment_ratio(cand, fish) >= pos_overlap_ratio and containment_ratio(fish, cand) >= pos_minor_containment_ratio
        is_mostly_fish = lambda cand, fish: containment_ratio(fish, cand) >= pos_containment_ratio and containment_ratio(cand, fish) >= pos_minor_overlap_ratio

    print("Cropping %s input images to %s..." % (len(imgs), settings.CROPS_OUTPUT_DIR))

    classes_encountered = []
    n = 0
    n_img = 0
    crop_file_name_keys = {}

    for img, meta in zip(imgs, metas):
        # Load image
        img = img()

        n_img += 1
        if n_img % 100 == 0:
            print('Cropped %d images...' % n_img)
        
        if metastr not in meta:
            # No bounding boxes/candidates, so skip this image
            continue

        if not no_histogram_matching:
            if colour.is_night_vision(img):
                # We are performing histogram matching and this image is night vision,
                # so histogram match it
                img = preprocessing.hist_match(img, template)
        
        if not ground_truth:
            if 'bounding_boxes' in meta:
                ref_bboxes = meta['bounding_boxes']
            else:
                ref_bboxes = []
        
        # For each crop...
        cand_bboxes = meta[metastr]
        bboxes = cand_bboxes + preprocessing.random_negative_boxes(img, cand_bboxes, num_FPs)
        crops = zip(*preprocessing.crop(img, bboxes, zoom_factor=zoom_factor))
        
        for i in range(len(bboxes)):
            cand_bbox = bboxes[i]
            n += 1
            
            if ground_truth or dataset == 'train':
                crop, clss = next(crops)
            else:
                crop = next(crops)
            
            if ground_truth:
                # Cropping the ground truth bounding boxes
                outcls = clss
            else:
                # Cropping the candidate regions
                if dataset == 'train':
                    matching_fish = [fish for fish in ref_bboxes if contains_most_of_fish(cand_bbox, fish) or is_mostly_fish(cand_bbox, fish)]
                    if len(matching_fish) > 0:
                        outcls = matching_fish[0]['class']
                    elif all(containment_ratio(preprocessing.zoom_box(cand_bbox, img.shape, output_dict=True), fish) <= neg_overlap_ratio for fish in ref_bboxes): # negative even when zoomed out
                        outcls = "NoF"
                    else: # too ambiguous for fish-or-not training data
                        continue
                else:
                    outcls = "test"

            
            file_name = "img_%s.jpg" % n
            class_dir = os.path.join(settings.CROPS_OUTPUT_DIR, outcls)
            file_path = os.path.join(class_dir, file_name)

            file_name_part = os.path.splitext(file_name)[0]

            crop_file_name_keys[file_name_part] = meta['filename']

            # Create directory for class if it does not exist yet
            if outcls not in classes_encountered:
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                classes_encountered.append(outcls)

            # Save the crop to file
            imsave(file_path, crop)
    
    if len(crop_file_name_keys) > 0:
        with open(os.path.join(settings.CROPS_OUTPUT_DIR, "_keys.json"), 'w') as outfile:
            json.dump(crop_file_name_keys, outfile)
    else:
        print('No region specification file found; are your JSONs in the right place?')
    
    print("All images cropped.")


if __name__ == "__main__":
    run(example,
        example_train_and_validation_split,
        example_crop_plot,
        example_augmentation,
        #
        train_network,
        #
        train_top_xception_network,
        fine_tune_xception_network,
        #
        train_top_vgg_network,
        fine_tune_vgg_network,
        #
        train_top_resnet_network,
        fine_tune_resnet_network,
        #
        train_top_localizer_vgg16_network,
        fine_tune_localizer_vgg16_network,
        #
        train_top_fish_or_no_fish_network,
        fine_tune_fish_or_no_fish_network,
        #
        train_top_fish_or_no_fish_resnet_network,
        fine_tune_fish_or_no_fish_resnet_network,
        #
        train_top_ground_truth_fish_or_no_fish_resnet_network,
        fine_tune_ground_truth_fish_or_no_fish_resnet_network,
        #
        train_top_fish_or_no_fish_vgg_network,
        fine_tune_fish_or_no_fish_vgg_network,
        #
        example_fully_convolutional,
        #
        segment_dataset,
        convert_annotations_to_darknet,
        crop_images)
