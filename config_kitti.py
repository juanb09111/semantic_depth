# All dirs relative to root
BATCH_SIZE = 1

# MODEL =  "EfusionPS_V3_depth" 
MODEL = "FuseNet"
MODEL_WEIGHTS_FILENAME_PREFIX = "EfusionPS_V3_depth"

BACKBONE = "resnet50" # This is the only one available at the moment
BACKBONE_OUT_CHANNELS = 256
NUM_THING_CLASSES = 3 #excluding background
NUM_STUFF_CLASSES = 12 #excluding background

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
MAX_TRAINING_SAMPLES = 10
VAL_SIZE = 0.20 #PERCENTAGE
# If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
USE_PREEXISTING_DATA_LOADERS = False
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/vkitti_data_loader_train_all_samples.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/vkitti_data_loader_val_all_samples.pth"


COCO_ANN = "kitti2coco_ann.json"


# ----------INFERENCE ----------

TEST_DIR = "data_val/"

INSTANCE =  False
SEMANTIC =  False
PANOPTIC =  True

# --------EVALUATION---------------

# Set the model weights to be used for evaluation
# MODEL_WEIGHTS_FILENAME = "tmp/models/FuseNet_weights_loss_0.11186084686504852.pth"
# MODEL_WEIGHTS_FILENAME = "tmp/models/FuseNet_weights_loss_0.30936820443942586.pth"
MODEL_WEIGHTS_FILENAME = "tmp/models/EfusionPS_V3_depth_loss_0.6401694336146243.pth"
# MODEL_WEIGHTS_FILENAME = "tmp/models/EfusionPS_V3_Blocks=12_depth_branch/EfusionPS_V3_depth_loss_0.44096572112988053.pth"
# Set the data loader to be used for evaluation. This can be set to None to use default filename
DATA_LOADER = None


# ----------INFERENCE ----------

# TEST_DIR = "kitti_video/"
TEST_DIR = "data_kitti/kitti_depth_completion_unmodified/"
# TEST_DIR = "data_jd/data_jd/"

# -------- REAL TIME ------

# Camera source can be either a camera device or a video
# CAM_SOURCE = 0
CAM_SOURCE = "plain.avi"
SAVE_VIDEO = True
RT_VIDEO_OUTPUT_BASENAME = "rt"
RT_VIDEO_OUTPUT_FOLDER = "rt_videos/kitti_depth_completion/"

# ----- MAKE VIDEO --------

FPS = 5

# folder containing the images
VIDEO_CONTAINER_FOLDER = "images_video/"
# video filename
VIDEO_OUTOUT_FILENAME = "video_name.avi"