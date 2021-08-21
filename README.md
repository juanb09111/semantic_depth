# EfussionPS
EfficientPS + FuseNet

# Requirements

* Linux (tested on Ubuntu 18.04)
* conda 4.7+
* CUDA 9.0 or higher

Other requirements such as pytorch, torchvision and/or cudatoolkit will be installed when creating the conda environment from the yml file.

# Getting Started

## Init

Create temporal folders and image directories. Run from terminal:
```
./init.sh
```

## Create conda virtual environment

```
conda env create -f conda_environment.yml
```


# Data

* Put your coco annotations file under the root folder
* Place all the training images  under data/

Set AUTOMATICALLY_SPLIT_SETS = True in config.py the first time you run train_ignite.py  to populate data_train and data_val folders with random images taken from the data/ folder. The validation set size can be set in config.py with SPLITS.VAL_SIZE. 
If you want to keep the same training and validation sets for future training runs set AUTOMATICALLY_SPLIT_SETS = False in config.py

```
Pynoptorch
├── data
├── data_train (Automatically created)
├── data_val (Automatically created)
├── semantic_segmentation_data (put all the semantic segmentation masks here)
├── augmented_data (put all augmented/stylized images here)
├── coco_hasty_annotations.json
.
.
.
```

# Train with ignite 

First of all, set up your configuration file config.py. There you will be able to select your network backbone, amount of evaluation images, pre-trained weights and more.

After having organized your data and set up your config file, simply run:

```
python train_ignite.py
```

The weights are saved every epoch under tmp/models/ and the progress is saved in tmp/res/training_results.txt

# Evaluate

Set MODEL_WEIGHTS_FILENAME in config.py  eg,. "tmp/models/EfficientPS_weights_maskrcnn_backbone_loss_0.57_bbx_82.8_segm_72.0.pth". Then run:

```
python eval_coco.py
```

# Inference 

To do inference simply put the images you want to do inference on under the folder {config.TEST_DIR}. Results will be saved in folders named according to the following pattern:
```
${MODEL_NAME}_${BACKBONES_NAME}_results_${SEGMENTATION_TYPE}_${TIMESTAMP}
```
Those folder are created under results/

# Make video out of inference

To make a video out of subsequent images simply spicify the folder containing the images and the name of output .avi file in config.VIDEO_CONTAINER_FOLDER
and config.VIDEO_OUTOUT_FILENAME respectively. Then run:

```
python make_video.py
```

# TODO:

* Create a Docker container 
<!-- # Commit

To commit to this repository please follow smart commit syntax: https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/ -->
