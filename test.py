# %%
# import segmentation_heads.roi_heads
# import backbones_bank.efficient_ps_backbone
# import segmentation_heads.RPN
# import models_bank.efficient_ps
# import common_blocks.threshold_instances
# import utils.make_video
# import utils.get_splits as get_splits
# import backbones_bank.tunned_maskrcnn.mask_rcnn
# import models_bank.efficient_ps2
# import depth_completion.project_lidar_2_cam
# import models_bank.fuseblock_2d_3d
# import utils.read_rosbag 
# import utils.get_kitti_dataset
# import utils.get_vkitti_dataset
# import utils.get_vkitti_dataset_full
# import utils.data_loader_2_coco_ann
import utils.kitti2coco
# import utils.kitti2coco_crop
# import utils.get_kitti_depth_gt

# get_splits.get_splits()
# %%
# import torch
# import config
# from utils.tensorize_batch import tensorize_batch
# import matplotlib.pyplot as plt
# data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

# # iterator = iter(data_loader_val)

# for _, anns in data_loader_val:
#     print(list(map(lambda ann: ann["category_ids"], anns)))


# images, anns = next(iterator)

# semantic_masks = list(map(lambda ann: ann['semantic_mask'], anns))
# semantic_masks = tensorize_batch(semantic_masks)
# semantic_masks = semantic_masks.long()
# sample_mask = semantic_masks[0]
# print(sample_mask.max())
# plt.imshow(sample_mask)

# import cv2
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import config
# import constants
# from PIL import Image
# import numpy as np
# import os.path

# from utils import get_datasets
# #%% Test reading images
# img = Image.open('semantic_segmentation_data/FrvLog1-14462-0-1548839592.9608.jpg.png')
# img = np.array(img)
# # img = cv2.imread('semantic_segmentation_data/FrvLog1-14462-0-1548839592.9608.jpg.png')
# img_tensor = torch.as_tensor(img, dtype=torch.uint8)
# plt.imshow(img)
# #%% Test dataloaders

# train_dir = os.path.join(os.path.dirname(
#             os.path.abspath(__file__)), constants.TRAIN_DIR)

# val_dir = os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), constants.VAL_DIR)


# train_ann_filename = os.path.join(os.path.dirname(
#             os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_TRAIN_DEFAULT_NAME)
# val_ann_filename = os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME)

# coco_ann_train = os.path.join(os.path.dirname(
#             os.path.abspath(__file__)), train_ann_filename)

# coco_ann_val = os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), val_ann_filename)

# data_loader_train = get_datasets.get_dataloaders(
#             config.BATCH_SIZE, train_dir, coco_ann_train, semantic_masks_folder='semantic_segmentation_data')

# data_loader_val = get_datasets.get_dataloaders(
#     config.BATCH_SIZE, val_dir, coco_ann_val, semantic_masks_folder='semantic_segmentation_data')

# iterator = iter(data_loader_val)

# images, anns = next(iterator)

# %%

# import json
# import numpy as np


# def blabla(res_json, ann_json):

#     with open(res_json, "r") as res_file:
#         res_data = json.load(res_file)

#     with open(ann_json, "r") as ann_file:
#         ann_data = json.load(ann_file)

#     res_image_ids = list(map(lambda x: x["image_id"], res_data))
#     ann_image_ids = list(map(lambda x: x["id"], ann_data["images"]))

#     res_image_ids = np.unique(res_image_ids)
#     print(len(res_image_ids), len(ann_image_ids))

# blabla("tmp/res/instances_val_obj_results.json",
#        "tmp/coco_ann/coco_ann_val_obj.json")

# import json 
# def get_id_list_res(ann_file):
#     ids = None
#     with open(ann_file) as ann:
#         data = json.load(ann)
#         ids = list(map(lambda ann: ann["image_id"], data))
#     checked_ids = []
#     with open("res_ids.txt", "w+") as ids_text:
#         for img_id in ids:
#             if img_id not in checked_ids:
#                 checked_ids.append(img_id)
#                 ids_text.write(str(img_id)+"\n")

# def get_id_list_ann(ann_file):
#     ids = None
#     with open(ann_file) as ann:
#         data = json.load(ann)
#         ids = list(map(lambda img: img["id"], data["images"]))
#     checked_ids = []

#     with open("ann_ids.txt", "w+") as ids_text:
#         for img_id in ids:
#             if img_id not in checked_ids:
#                 checked_ids.append(img_id)
#                 ids_text.write(str(img_id)+"\n")

# get_id_list_res("tmp/res/instances_val_obj_results.json")
# get_id_list_ann("tmp/coco_ann/coco_ann_val_obj.json")
