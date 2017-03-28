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

# Running

The main functionality of the project is accessed through the `main.py` script. For an overview of available commands, run:

`python src/main.py --help`

## Using the project to train from scratch

To start using the network from scratch, follow these steps:

### Download
Download the competition data and the bounding boxes (todo: give link).

### Prepare the directory structure

Place the competitino data and bounding boxes into a directory structure as follows (or edit `settings.py` so it points to the correct location).

```
+-- data
|   +-- bounding_boxes
    |   +-- ALB.json
    |   +-- BET.json
    |       ...
|   +-- train
    |   +-- ALB
    |   +-- BET
    |       ...
+-- output
+-- src
|   +-- __init__.py
|   +-- colour.py
|   +-- darknet.py
    ...
```

### Prepare the image crops
Prepare the crops, including colorization through histogram matching for the "night vision" images:

```
python src/main.py crop-image
```

This will take roughly half an hour to complete. Note that multiple crops can be generated per input image, if multiple fish are present. The images are placed in `output/crops/_timestamp_/`. Once the process is complete, move the generated crops to `data/cropped_train`, such that it has the same directory structure as `data/train`.

### Transfer-learn a new network based on Xception

Run the following to create a new model based on a pretrained Xception network. The new layers are trained:

```
python src/main.py train-top-xception-network
```

This saves weights in `data/weights`.

### Fine-tune the transfer-learned network

Run the following to fine-tune the trained network, this also trains some of the layers that had been pre-trained. (These steps are separated, as if we were to immediately update all layers, the pretrained layers would be updated with essentially random gradients).

```
python src/main.py fine-tune-xception-network
```

The weights are saved in `data/weights`.