# Fisheries monitoring

A machine learning project for the [Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) Kaggle competition.

# Requirements

Developed with:

- Python 3.4 or 3.5
- Numpy 1.12.1
- Scipy: 0.19.0
- Scikit-image 0.12.3
- Scikit-learn 0.18.1 
- Keras: 2.0.1
- Tensorflow: 1.0.1

The project might work with some newer or later versions of these packages as well.

# Data and directory structure

Download the competition data and the bounding boxes (todo: give link).

Place the competition data and bounding boxes into a directory structure as follows (or edit `settings.py` so it points to the correct location). 

**Initially you will only have `bounding_boxes` and `train` available to put inside `data`.** The other directories (such as `cropped_candidates`) will be filled using output of the various model stages themselves.

```
+-- data
|   +-- train
|   |   +-- original
|   |   |   +-- ALB
|   |   |   |   +-- img_00003.jpg
|   |   |   |       ...
|   |   |   +-- BET
|   |   |       ...
|   |   +-- bounding_boxes_ground_truth
|   |   |   +-- ALB.json
|   |   |   +-- BET.json
|   |       ...
|   |   +-- bounding_boxes_candidates
|   |   |   +-- ALB_candidates_0-x.json
|   |   |   +-- ALB_candidates_x-y.json
|   |   |       ...
|   |   +-- cropped_ground_truth
|   |   |   +-- ALB
|   |   |   |   +-- img_1.jpg
|   |   |   |       ...
|   |   |   +-- BET
|   |   |       ...
|   |   +-- cropped_candidates
|   |   |   +-- negative
|   |   |   |   +-- img_110.jpg
|   |   |   |       ...
|   |   |   +-- positive
|   +-- test
|   |   +-- original
|   |   |   +-- img_00001.jpg
|   |   |       ...
|   |   +-- bounding_boxes_candidates
|   |   |   +-- candidates_0-x.json
|   |   |   +-- candidates_x-y.json
|   |   |       ...
|   |   +-- cropped_candidates
|   |   |   +-- img_1.jpg
|   |   |       ...
|   +-- weights
|   |   +-- fishnofish.ext_xception.toptrained.e001-tloss0.3366-vloss0.2445.hdf5
|   |       ...
+-- output
+-- src
|   +-- __init__.py
|   +-- colour.py
|   +-- darknet.py
|       ...
```

# Running

The main functionality of the project is accessed through the `main.py` script. For an overview of available commands, run:

`python src/main.py --help`

# Preparing data

## Prepare ground truth bounding box image crops
Prepare the training image crops, including colorization through histogram matching for the "night vision" images:

```
python src/main.py crop-images train --crop-type=ground_truth
```

This will take roughly half an hour to complete. Note that multiple crops can be generated per input image, if multiple fish are present. The images are placed in `output/crops/_timestamp_/`. Once the process is complete, move the generated crops to `data/train/cropped_ground_truth`.

## Prepare fish region candidates

Run the following to prepare the fish region candidates for the training set:

```
python src/main.py segment-dataset train
```

This will take a long time to complete (~1 day).

Optionally a range of images can be given which have to be segmented:

```
python src/main.py segment-dataset train 0-99
python src/main.py segment-dataset train 100-199
```

Using this, multiple processes can be launched in parallel to segment different images.

## Prepare fish region candidate crops

Prepare the fish region candidate crops, including colorization through histogram matching for the "night vision" images:

```
python src/main.py crop-images train --crop-type=candidates
```

Many crops will be generated per image, and will be labeled either "negative" or "positive" using the ground truth bounding boxes. The images are placed in `output/crops/_timestamp_/`. Once the process is complete, move the generated crops to `data/train/cropped_candidates`.


# Using the project to train a fish classifier from scratch

To start training a fish classifier network using bounding boxes for fish in each image from scratch, follow these steps:

## Transfer-learn a new network based on Xception

Run the following to create a new model based on a pretrained Xception network. The new layers are trained:

```
python src/main.py train-top-xception-network
```

This saves weights in `output/weights/_timestamp_`. Copy the weights to `data/weights`.

## Fine-tune the transfer-learned network

(TODO: make this easier to work with). Edit `fine_tune_xception_network` in `main.py`: set `input_weights_name` to the file name of the weights copied in the previous step.

Run the following to fine-tune the trained network, this also trains some of the layers that had been pre-trained. (These steps are separated, as if we were to immediately update all layers, the pretrained layers would be updated with essentially random gradients).

```
python src/main.py fine-tune-xception-network
```

The weights are saved in `data/weights`.
