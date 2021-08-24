# Semantic Segmentation and Depth completion 
Semantic segmentation + FuseNet with shared backbone

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

*Download virtual kitti dataset

# Train with ignite 


After having organized your data and set up your config file, simply run:

*For depth completion only
```
python train_fusenet_vkitti.py
```
*For semantic segmentation only
```
python train_sem_seg_net.py
```
*For semantic segmentation and depth
```
python train_semseg_depth_vkitti.py
```

The weights are saved every epoch under tmp/models/ and the progress is saved in tmp/res/training_results.txt

```

# TODO:

* Create a Docker container 
<!-- # Commit

To commit to this repository please follow smart commit syntax: https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/ -->
