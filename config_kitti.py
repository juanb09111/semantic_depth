# All dirs relative to root
BATCH_SIZE = 1

# MODEL =  "EfusionPS_V3_depth" 
MODEL = "SemsegNet"
# CHECKPOINT = "tmp/models/FuseNet_loss_0.42496299164789214.pth"
CHECKPOINT = None

BACKBONE = "resnet50" # This is the only one available at the moment
BACKBONE_OUT_CHANNELS = 256
NUM_THING_CLASSES = 3 #excluding background
NUM_STUFF_CLASSES = 11 #excluding background

SEMANTIC_HEAD_DEPTHWISE_CONV = False
BACKBONE_DEPTHWISE_CONV = False

ORIGINAL_INPUT_SIZE_HW = (200, 1000)
# ORIGINAL_INPUT_SIZE_HW = (200, 1000)
RESIZE = 0.5
CROP_OUTPUT_SIZE = (200, 1000)
MIN_SIZE = 200 
MAX_SIZE = 1200 

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
MAX_TRAINING_SAMPLES = None
VAL_SIZE = 0.20 #PERCENTAGE
# If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
USE_PREEXISTING_DATA_LOADERS = True
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/vkitti_data_loader_train_all_samples.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/vkitti_data_loader_val_all_samples.pth"

COCO_ANN = "kitti2coco_ann.json"

# --------EVALUATION---------------
MODEL_WEIGHTS_FILENAME = "tmp/models/EfusionPS_V3_depth_loss_0.6401694336146243.pth"
DATA_LOADER = None

