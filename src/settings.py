"""
Module containing the project settings.
"""

import os
from time import strftime

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_BOUNDING_BOXES_DIR = os.path.join(DATA_DIR, "bounding_boxes")
TEST_DIR = os.path.join(DATA_DIR, "test")

CONVERTED_BOUNDING_BOX_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "bounding_boxes", strftime("%Y%m%dT%H%M%S"))

CLASS_NAME_TO_INDEX_MAPPING = {
    "ALB": 0,
    "BET": 1,
    "DOL": 2,
    "LAG": 3,
    "SHARK": 4,
    "YFT": 5,
    "OTHER": 6
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