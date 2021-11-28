# All dirs relative to root


BACKBONE = "resnet50" # This is the only one available at the moment
BACKBONE_OUT_CHANNELS = 256
NUM_THING_CLASSES = 3 #excluding background
NUM_STUFF_CLASSES = 11 #excluding background

SEMANTIC_HEAD_DEPTHWISE_CONV = False
BACKBONE_DEPTHWISE_CONV = False
PRE_TRAINED_BACKBONE = True

RESIZE = 0.5
CROP_OUTPUT_SIZE = (200, 700)
MIN_SIZE = 200  
MAX_SIZE = 700 

# for k-nn
K_NUMBER = 9
# number of 3D points for the model
N_NUMBER = 8000
# N_NUMBER = 4000
MAX_DEPTH = 50 # distance in meters
# alpha parameter for loss calculation
LOSS_ALPHA = 0.8


DATA = "data/"
RGB_DATA = "vkitti_2.0.3_rgb"
INSTANCE_SEGMENTATION_DATA = "vkitti_2.0.3_instanceSegmentation"
SEMANTIC_SEGMENTATION_DATA = "vkitti_2.0.3_classSegmentation" 
SEMANTIC_SEGMENTATION_DATA_CLASS = "semseg_bin" 
# DATA = "data_jd/data_jd/"

MAX_EPOCHS = 100

# MAX_TRAINING_SAMPLES = 100
MAX_TRAINING_SAMPLES = 100
VAL_SIZE = 0.20 #PERCENTAGE
# If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
USE_PREEXISTING_DATA_LOADERS = False
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/vkitti_data_loader_train_625_samples.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/vkitti_data_loader_val_625_samples.pth"

COCO_ANN = "kitti2coco_ann_crop.json"

# --------EVALUATION---------------
MODEL_WEIGHTS_FILENAME = "tmp/models/EfusionPS_V3_depth_loss_0.6401694336146243.pth"
DATA_LOADER = None
IOU_TYPES = ["bbox", "segm"]


# ----- INFERENCE-----------------


# # Object tracking
INFERENCE_DATA = "data/"
# INFERENCE_DATA = "data_vkitti_video"
OBJECT_TRACKING = False
# COCO_ANN_INFERNECE = "kitti_video_2_coco_ordered.json"
COCO_ANN_INFERNECE = "kitti2coco_ann_crop.json"
OUTPUT_FOLDER = "Panoptic_seg_TRAIN_crop_120s_200_1300"

MAX_DETECTIONS = 50 # Maximum number of tracked objects
NUM_FRAMES = 5 # Number of frames before recycling the ids

# Make video

# VIDEO_CONTAINER_FOLDER = "data_vkitti_video/vkitti_2.0.3_Panoptic/Scene20/morning/frames/rgb/Camera_0"
VIDEO_CONTAINER_FOLDER = "results/Panoptic_seg_rt"
VIDEO_OUTOUT_FILENAME = "Panoptic_seg_rt.avi"
FPS = 5