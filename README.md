# SemSegDepth, Segmentation and Depth completion 
SemSegDepth: A Combined Model for Semantic Segmentation and
Depth Completion

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

## 

# Data

*Download virtual kitti dataset from https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

# Train with ignite 


After having organized your data and set up your config file, simply run:


```
./train.sh Semseg_Depth_v4 2
```


The weights are saved every epoch under tmp/models/ and the progress is saved in tmp/res/results_Semseg_Depth_v4.txt



# Inference

```
./inference.sh Semseg_Depth_v4 2 tmp/models/weights_file.pth  COCO.json  your_dst_folder source_folder
```

<!-- # Commit

To commit to this repository please follow smart commit syntax: https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/ -->
