"""
Module containing the project settings.
"""

import os
from time import strftime

# Directory settings

#ROOT_DIR = '/vol/tensusers/vgarciacazorla/fm'
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

## Input directories

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

TRAIN_ORIGINAL_IMAGES_DIR                = os.path.join(TRAIN_DIR, "original")
TRAIN_GROUND_TRUTH_CROPPED_IMAGES_DIR    = os.path.join(TRAIN_DIR, "cropped_ground_truth")
TRAIN_CANDIDATES_CROPPED_IMAGES_DIR      = os.path.join(TRAIN_DIR, "cropped_candidates")
TRAIN_GROUND_TRUTH_BOUNDING_BOXES_DIR    = os.path.join(TRAIN_DIR, "bounding_boxes_ground_truth")
TRAIN_CANDIDATES_BOUNDING_BOXES_DIR      = os.path.join(TRAIN_DIR, "bounding_boxes_candidates")
TRAIN_FISH_OR_NO_FISH_CLASSIFICATION_DIR = os.path.join(TRAIN_DIR, "fish_or_no_fish_classification")
TRAIN_FISH_TYPE_CLASSIFICATION_DIR       = os.path.join(TRAIN_DIR, "fish_type_classification")

TEST_ORIGINAL_IMAGES_DIR                 = os.path.join(TEST_DIR, "original")
TEST_CANDIDATES_CROPPED_IMAGES_DIR       = os.path.join(TEST_DIR, "cropped_candidates")
TEST_CANDIDATES_BOUNDING_BOXES_DIR       = os.path.join(TEST_DIR, "bounding_boxes_candidates")
TEST_FISH_OR_NO_FISH_CLASSIFICATION_DIR  = os.path.join(TEST_DIR, "fish_or_no_fish_classification")
TEST_FISH_TYPE_CLASSIFICATION_DIR        = os.path.join(TEST_DIR, "fish_type_classification")

WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
TENSORBOARD_LOGS_DIR = os.path.join(DATA_DIR, 'tb_logs')

## Output directories

CONVERTED_BOUNDING_BOX_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "bounding_boxes", strftime("%Y%m%dT%H%M%S"))
SEGMENTATION_CANDIDATES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "candidates", strftime("%Y%m%dT%H%M%S"))
CROPS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "crops", strftime("%Y%m%dT%H%M%S"))
WEIGHTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "weights", strftime("%Y%m%dT%H%M%S"))

## Imagenet metadata

IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet")
IMAGENET_CLSLOC_PATH = os.path.join(IMAGENET_DIR, "meta_clsloc.mat")

# Problem-specific settings

## Classes

CLASS_NAME_TO_INDEX_MAPPING = {
    "ALB":      0,
    "BET":      1,
    "DOL":      2,
    "LAG":      3,
    "SHARK":    4,
    "YFT":      5,
    "OTHER":    6,
    "NoF":      7
    }

"""CLASS_INDEX_TO_NAME_MAPPING = {
    0: "ALB",
    1: "BET",
    2: "DOL",
    3: "LAG",
    4: "SHARK",
    5: "YFT",
    6: "OTHER"
    }"""

CLASS_INDEX_TO_NAME_MAPPING = {v: k for k, v in CLASS_NAME_TO_INDEX_MAPPING.items()}
## Data augmentation
AUGMENTATION = {
    'agressive':{   
            'RESCALE':               1./255,
            'ROTATION_RANGE':         360,  
            'SHEAR_RANGE':            0.1,    
            'ZOOM_RANGE':             [0.6, 0.9],   
            'WIDTH_SHIFT_RANGE':      0.2,  
            'HEIGHT_SHIFT_RANGE':     0.2,
            'HORIZONTAL_FLIP':        True,
            'VERTICAL_FLIP':          True,
            'CHANNEL_SHIFT_RANGE':    20.0,
            'BLUR_RANGE':             [0., 1.5]},    
    'moderate':{   
            'RESCALE':                 1./255,
            'ROTATION_RANGE':          360,  
            'SHEAR_RANGE':            0.05,    
            'ZOOM_RANGE':             [0.9, 1],   
            'WIDTH_SHIFT_RANGE':      0.05,  
            'HEIGHT_SHIFT_RANGE':     0.05,
            'HORIZONTAL_FLIP':        True,
            'VERTICAL_FLIP':          True,
            'CHANNEL_SHIFT_RANGE':    15.0,
            'BLUR_RANGE':             [0., 1.]}}
                          

