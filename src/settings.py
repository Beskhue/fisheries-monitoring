"""
Module containing the project settings.
"""

import os
from time import strftime

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_BOUNDING_BOXES_DIR = os.path.join(DATA_DIR, "bounding_boxes")
TEST_DIR = os.path.join(DATA_DIR, "test")
