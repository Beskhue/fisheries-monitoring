import os
import pprint
import json
from clize import run, parameters
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
    
    def display_img_and_heatmap(img, heatmap):
        import matplotlib.pyplot as plt
        import scipy.misc

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(img.astype("uint8"))
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, interpolation='nearest', cmap="viridis")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        
        plt.imshow(img.astype("uint8"))
        heatmap_resized = scipy.misc.imresize(heatmap, img.shape)
        plt.imshow(heatmap_resized, interpolation='nearest', cmap="viridis", alpha=0.5)
        plt.axis('off')
        plt.show()

    import network

    netw = network.LearningFullyConvolutional()
    netw.build(weights_file = "classification.ext_resnet.toptrained.e003-tloss0.2131-vloss0.4505.hdf5", num_classes = 7)

    pl = pipeline.Pipeline(data_type = "original")

    generator = pl.data_generator_builder()


    for i in range(25):
        x, y, meta = next(generator)

        heatmap = netw.build_multi_scale_heatmap(x)
        display_img_and_heatmap(x, heatmap)

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

    tl = network.TransferLearning(data_type = "ground_truth_cropped", class_filter = ["NoF"])

    tl.build('xception', summary = False)
    tl.train_top(epochs = 70)

def fine_tune_xception_network():
    """
    Fine-tune the extended xception network. To do this, first the top
    of the extended xception network must have been trained already.
    """

    import network

    tl = network.TransferLearning8ClassCLF(data_type = "ground_truth_cropped", class_filter = ["NoF"])

    tl.build('xception', summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "ext_xception_toptrained.hdf5",
        n_layers = 125)

def train_top_xception_8_class_clf_on_candidates_network(epochs = 70):
    """
    Train the top of the extended xception network.
    """
    import network
    data_type = "candidates_cropped_8_classes"
    #data_type = "ground_truth_cropped"
    tl = network.TransferLearning8ClassCLF(data_type = data_type, class_filter = [], prediction_class_type = "multi", class_balance_method = "batch")
    tl.build('xception', summary = True)
    tl.train_top(epochs = epochs)

def fine_tune_xception_8_class_clf_on_candidates_network(layers_to_freeze_from_bottom = 125, epochs = 70):
    """
    Fine-tune the extended xception network. To do this, first the top
    of the extended xception network must have been trained already.
    """
    import network
    #data_type = "candidates_cropped_8_classes"
    data_type = "ground_truth_cropped"
    tl = network.TransferLearning8ClassCLF(data_type = data_type, class_filter = [], prediction_class_type = "multi", class_balance_method = "batch")

    tl.build('xception', summary = True)
    tl.fine_tune_extended(
        epochs = epochs,
        input_weights_name = 'top_trained_xception.hdf5',
        n_layers = layers_to_freeze_from_bottom)

def train_top_resnet_network():
    """
    import network
    Train the top of the extended resnet50 fish type classification network.
    """
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

    tl = network.TransferLearning(data_type = "candidates_cropped", class_filter = ["NoF"])

    tl.build('vgg19', summary = False)
    tl.train_top(epochs = 70)

def fine_tune_vgg_network():
    """
    Fine-tune the extended vgg19 network. To do this, first the top
    of the extended vgg19 network must have been trained already.
    """

    import network

    tl = network.TransferLearning(data_type = "candidates_cropped", class_filter = ["NoF"])

    tl.build('vgg19', summary = False)
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

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "candidates_cropped")

    tl.build('xception', summary = False)
    tl.train_top(epochs = 70)

def fine_tune_fish_or_no_fish_network():
    """
    Fine-tune the extended xception fish or no fish network. To do this, first the top
    of the extended xception network must have been trained already.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "candidates_cropped")

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

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "ground_truth_cropped")

    tl.build('resnet', input_shape = (300, 300, 3), summary = False)
    tl.train_top(epochs = 70)

def fine_tune_fish_or_no_fish_resnet_network():
    """
    Fine-tune the extended xception fish or no fish network. To do this, first the top
    of the extended xception network must have been trained already.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "ground_truth_cropped")

    tl.build('resnet', input_shape = (300, 300, 3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "fishnofish.ext_resnet.toptrained.e034-tloss0.2205-vloss0.2006.hdf5",
        n_layers = 25)

def train_top_fish_or_no_fish_vgg_network():
    """
    Train the top of the extended inceptionv4 fish or no fish network.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "candidates_cropped")
    
    tl.build('vgg19', input_shape = (300, 300, 3), summary = False)
    
    tl.train_top(epochs = 70)

def fine_tune_fish_or_no_fish_vgg_network():
    """
    Fine-tune the extended inceptionv4 fish or no fish network. To do this, first the top
    of the extended inceptionv4 network must have been trained already.
    """

    import network

    tl = network.TransferLearningFishOrNoFish(class_balance_method = "batch", prediction_class_type = "single", data_type = "candidates_cropped")

    tl.build('vgg19', input_shape = (300, 300, 3), summary = False)
    tl.fine_tune_extended(
        epochs = 70,
        input_weights_name = "ext_vgg19.toptrained.e060-tloss0.5310-vloss0.5097.hdf5",
        n_layers = 17)

def segment_dataset(dataset, index_range=None, *, silent=False):
    """
    Segments (part of) the given data set by colour, producing and saving candidate bounding boxes in a JSON file. Note: slow!
    
    dataset: Which image data to segment: train, test or final.
    
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
    segmentation.do_segmentation(img_idxs=index_range, output = not silent, save_candidates=True, data=dataset)

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
                crop_type : parameters.one_of(("candidates", "Create crops using the candidate regions."), ("ground_truth", "Create crops using the ground truth fish bounding boxes."), case_sensitive = True) = "candidates", 
                no_histogram_matching = False):
    """
    Crop images in the data using either bounding box annotations or generated candidates. Creates one crop for each bounding box / candidate.
    
    dataset: which data set to crop images from (train, test, final)

    crop_type: whether to use candidate regions or ground-truth bounding boxes for cropping
    
    no_histogram_matching: disable histogram matching to colour in night-vision images
    """

    import preprocessing
    from skimage.io import imsave

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
    else:
        metastr = "candidates"
        pos_overlap_ratio = 0.65
        pos_minor_overlap_ratio = 0.1
        pos_containment_ratio = 0.50
        pos_minor_containment_ratio = 0.17
        neg_overlap_ratio = 0.10
        
        intersection = lambda cand, fish: max(0, min(cand['x']+cand['width'], fish['x']+fish['width']) - max(cand['x'], fish['x'])) * max(0, min(cand['y']+cand['height'], fish['y']+fish['height']) - max(cand['y'], fish['y']))
        containment_ratio = lambda cand, fish: intersection(cand, fish) / float(fish['width']*fish['height'])

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
        crops = zip(*preprocessing.crop(img, meta[metastr]))
        for i in range(len(cand_bboxes)):
            cand_bbox = cand_bboxes[i]
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
                    box_x_low, box_x_high, box_y_low, box_y_high = preprocessing.zoom_box(cand_bbox, img.shape)
                    cand_crop = {'x': box_x_low, 'width': box_x_high-box_x_low, 'y': box_y_low, 'height': box_y_high-box_y_low}
                    
                    if any(containment_ratio(cand_bbox, fish) >= pos_overlap_ratio and containment_ratio(fish, cand_bbox) >= pos_minor_containment_ratio for fish in ref_bboxes): # more than 65% of a fish is inside
                        outcls = "positive"
                    elif any(containment_ratio(fish, cand_bbox) >= pos_containment_ratio and containment_ratio(cand_bbox, fish) >= pos_minor_overlap_ratio for fish in ref_bboxes): # more than half of this b.box contains a fish (for overly large sharks)
                        outcls = "positive"
                    elif all(containment_ratio(cand_crop, fish) <= neg_overlap_ratio for fish in ref_bboxes): # negative even when zoomed out
                        outcls = "negative"
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
        train_top_xception_8_class_clf_on_candidates_network,
        fine_tune_xception_8_class_clf_on_candidates_network,
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
        train_top_fish_or_no_fish_vgg_network,
        fine_tune_fish_or_no_fish_vgg_network,
        #
        example_fully_convolutional,
        #
        segment_dataset,
        convert_annotations_to_darknet,
        crop_images)
