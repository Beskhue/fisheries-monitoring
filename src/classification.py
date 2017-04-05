import math
from clize import run
from clize.parameters import argument_decorator
import os
import functools
import json
import numpy as np
import main
import settings
import pipeline

class ClassificationParams:
    """
    A class with the parameters needed to perform classification on a set of images, using various (pre-trained) parts.
    """

    def __init__(self, dataset = "test"):
        """
        :param dataset: "test" or "final"
        """
        self.dataset = dataset
        self.prepare_directories()

    def prepare_directories(self):
        if self.dataset == "train":
            self.fish_or_no_fish_classification_dir = settings.TRAIN_FISH_OR_NO_FISH_CLASSIFICATION_DIR
            self.fish_type_classification_dir = settings.TRAIN_FISH_TYPE_CLASSIFICATION_DIR
        elif self.dataset == "test":
            self.fish_or_no_fish_classification_dir = settings.TEST_FISH_OR_NO_FISH_CLASSIFICATION_DIR
            self.fish_type_classification_dir = settings.TEST_FISH_TYPE_CLASSIFICATION_DIR
        elif self.dataset == "final":
            pass # not yet implemented

@argument_decorator
def prep_classif(dataset):
    if dataset not in ["train", "test", "final"]:
        print("Unknown data set: " + dataset)
        exit()
    else:
        cp = ClassificationParams(dataset = dataset)
        return cp

def propose_candidates(params:prep_classif):
    """
    Stage 1
    """
    main.segment_dataset(params.dataset)

def crop_candidates(params:prep_classif):
    """
    Stage 2
    """
    main.crop_images(params.dataset, crop_type = "candidates")

def propose_candidates_fullyconv(params:prep_classif):
    """
    Alternative stage 1
    """
    main.segment_dataset(params.dataset, type="fullyconv")

def crop_candidates_fullyconv(params:prep_classif):
    """
    Alternative stage 2
    """
    main.crop_images(params.dataset, crop_type = "candidates_fullyconv")

def classify_fish_or_no_fish(params:prep_classif):
    """
    Stage 3
    """

    mini_batch_size = 32 # decrease in case of resource exhausted errors
    
    import keras
    import metrics
    import shutil
    from time import strftime
    
    ppl = pipeline.Pipeline(data_type = "candidates_fullyconv_cropped", dataset = params.dataset)

    # Load fish-or-no-fish classification model
    model = keras.models.load_model(os.path.join(settings.WEIGHTS_DIR, settings.FISH_OR_NO_FISH_CLASSIFICATION_NETWORK_WEIGHT_NAME), custom_objects={'precision': metrics.precision, 'recall': metrics.recall})

    data_generator = ppl.data_generator_builder(
        functools.partial(ppl.mini_batch_generator, mini_batch_size = mini_batch_size))

    predicted = {}
    n_batches = 0
    batch_print_interval = int(100/mini_batch_size)+1

    # For each batch
    for x, y, meta in data_generator:
        x = np.array(x)

        predictions = model.predict(x, batch_size = mini_batch_size)

        for m, pred in zip(meta, list(predictions)):
            predicted[m['filename']] = float(pred[0])
        
        n_batches += 1
        if n_batches % batch_print_interval == 0:
            print('%d candidates processed' % (mini_batch_size*n_batches))

    # Save classifications
    if not os.path.exists(params.fish_or_no_fish_classification_dir):
        os.makedirs(params.fish_or_no_fish_classification_dir)
    
    outpath = os.path.join(params.fish_or_no_fish_classification_dir, "classification.json")
    outpath2 = os.path.join(params.fish_or_no_fish_classification_dir, "classification-%s.json" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath, 'w') as outfile:
        json.dump(predicted, outfile)
    
    shutil.copyfile(outpath, outpath2)

def classify_fish_type(params:prep_classif):
    """
    Stage 4
    """

    import keras
    import metrics
    import shutil
    from time import strftime
    
    threshold = 0.5

    ppl = pipeline.Pipeline(data_type = "candidates_fullyconv_cropped", dataset = params.dataset)

    # Load fish type classification model
    model = keras.models.load_model(os.path.join(settings.WEIGHTS_DIR, settings.FISH_TYPE_CLASSIFICATION_NETWORK_WEIGHT_NAME), custom_objects={'precision': metrics.precision, 'recall': metrics.recall})

    data = ppl.get_data()

    fish_type_classification = {}
    n_imgs = 0

    # For each single crop
    for x, meta in zip(data['x'], data['meta']):
        #x = np.array(x)


        img = x()
        img = np.array([img])

        predictions = model.predict(img, batch_size = 1)
            
        fish_type_classification[meta['filename']] = [float(pred) for pred in predictions.tolist()[0]]
        
        n_imgs += 1
        if n_imgs % 100 == 0:
            print('%d candidates processed' % n_imgs)

    # Save classifications
    if not os.path.exists(params.fish_type_classification_dir):
        os.makedirs(params.fish_type_classification_dir)
    
    outpath = os.path.join(params.fish_type_classification_dir, "classification.json")
    outpath2 = os.path.join(params.fish_type_classification_dir, "classification-%s.json" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath, 'w') as outfile:
        json.dump(fish_type_classification, outfile)
    
    shutil.copyfile(outpath, outpath2)

def classify_image(params:prep_classif):
    """
    Stage 5
    """
    
    import csv
    import math
    import shutil
    from time import strftime

    threshold = 0.5
    
    # Load fish type classifications
    inpath = os.path.join(params.fish_type_classification_dir, "classification.json")
    with open(inpath, 'r') as infile:
        fish_type_classification = json.load(infile)

    ppl = pipeline.Pipeline(data_type = "candidates_fullyconv_cropped", dataset = params.dataset)
    data = ppl.get_data()

    # Load fish-or-no-fish classifications
    inpath = os.path.join(params.fish_or_no_fish_classification_dir, "classification.json")
    with open(inpath, 'r') as infile:
        fish_or_no_fish = json.load(infile)

    cand_classifications = {}
    # Aggregate classifications of bounding boxes to original image level
    for meta in data['meta']:
        name = meta['filename']

        if fish_or_no_fish[name] > threshold:
            
            if name in fish_type_classification:
                # There is a classification for this crop
                original_img = meta['original_image']
    
                if original_img not in cand_classifications:
                    cand_classifications[original_img] = []

                cand_classifications[original_img].append({'p_fish': fish_or_no_fish[name], 'p_fish_types': fish_type_classification[name]})
            
    outpath2 = os.path.join(params.fish_type_classification_dir, "aggregatedclassification-%s.json" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath2, 'w') as outfile:
        json.dump(cand_classifications, outfile)
    
    # Perform something to turn list of classifications for an image to scores for all classes
    img_classifications = {}

    pipeline_original = pipeline.Pipeline(data_type = "original", dataset = params.dataset)
    original_images = pipeline_original.get_data()
    for meta in original_images['meta']:
        name = meta['filename']
        if name not in cand_classifications: # no candidates are proposed (or no proposed candidates are accepted by the fish-or-no-fish ensemble)
            print('Image has zero detected fish candidates: ' + name)
            #                            ALB   BET   DOL   LAG   NoF   OTHER SHARK YFT
            img_classifications[name] = [0.001,0.001,0.001,0.001,0.993,0.001,0.001,0.001]
        else:

            """
            # We grab the prediction of the candidate where the main classification is assigned the most confidence.
            # Note that a most-confident NoF classification is never chosen, unless all candidates have NoF as the
            # most-confident prediction.
            nof_idx = 4
            most_certain_non_nof = 0
            idx_most_certain_non_nof = None
            most_certain_nof = 0
            idx_most_certain_nof = None

            idx_cand = 0
            for cand_classification in cand_classifications[name]:

                # Find index of the predicted fish type (i.e., the type with the highest probability according to the network)
                idx_max = np.argmax(cand_classification)

                if idx_max == nof_idx:
                    # The predicted type is NoF

                    if cand_classification[nof_idx] > most_certain_nof:
                        most_certain_nof = cand_classification[nof_idx]
                        idx_most_certain_nof = idx_cand
                else:
                    # The predicted type is not NoF

                    if cand_classification[idx_max] > most_certain_non_nof:
                        most_certain_non_nof = cand_classification[idx_max]
                        idx_most_certain_non_nof = idx_cand

                idx_cand += 1
            
            if idx_most_certain_non_nof != None:
                img_classifications[name] = cand_classifications[name][idx_most_certain_non_nof]
            else:
                img_classifications[name] = cand_classifications[name][idx_most_certain_nof]
            """

            # Find the NoF predictions

            nof_confidences = []

            for cand_classification in cand_classifications[name]:
                p_fish_types = cand_classification['p_fish_types']
                nof_confidences.append(p_fish_types[4])

            epsilon = 0.00000001

            #                 ALB  BET  DOL  LAG  OTHER SHARK YFT
            classification = [0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0]

            for cand_classification in cand_classifications[name]:
                p_fish = cand_classification['p_fish']
                p_fish_types = cand_classification['p_fish_types']

                classification = [a + math.log(b * p_fish) for a, b in zip(classification, p_fish_types[:4] + p_fish_types[5:])]

            classification = list(map(lambda a: a - max(classification), classification))
            classification = list(map(lambda a: math.exp(a), classification))
            classification = list(map(lambda a: a / sum(classification), classification))

            #                     ALB  BET  DOL  LAG  NoF OTHER SHARK YFT
            img_classification = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0]
            img_classification[0] = classification[0]
            img_classification[1] = classification[1]
            img_classification[2] = classification[2]
            img_classification[3] = classification[3]
            img_classification[4] = min(nof_confidences) # Least confident NoF predicition
            img_classification[5] = classification[4]
            img_classification[6] = classification[5]
            img_classification[7] = classification[6]

            img_classification = list(map(lambda a: a / sum(img_classification), img_classification))

            img_classifications[name] = img_classification
            
    # Output in kaggle format
    class_order = [0, 1, 2, 3, 4, 5, 6, 7]
    outpath = os.path.join(params.fish_type_classification_dir, "submission.json")
    outpath2 = os.path.join(params.fish_type_classification_dir, "submission-%s.json" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath, 'w', newline='') as subm_file:
        subm_writer = csv.writer(subm_file)
        subm_writer.writerow(['image', 'ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
        
        for meta in original_images['meta']:
            name = meta['filename']
            img_classification = img_classifications[name]
            img_classification = [img_classification[i] for i in class_order] # permute into Kaggle order
            subm_writer.writerow([name + '.jpg'] + img_classification)
    
    shutil.copyfile(outpath, outpath2)


if __name__ == "__main__":
    run(propose_candidates,
        crop_candidates,
        classify_fish_or_no_fish,
        classify_fish_type,
        classify_image)
