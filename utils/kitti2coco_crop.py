from torchvision.utils import save_image
import torch
import os.path
import math
from PIL import Image
import numpy as np
import config_kitti
import glob
import random
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
import json
from datetime import datetime

import torchvision.transforms as transforms
# TODO:
# get list of rgb images files
# get list of semantic segmentation files
# get list of instance segmentation files
# get unique values from instance segmentation
# map unique values to pixels
# find corresponding pixels in semantic segmentation
# map value to class
# write json file

rgb_2_class = [
    ("Terrain", [210, 0, 200], "background", 1),
    ("Sky", [90, 200, 255], "background", 2),
    ("Tree", [0, 199, 0], "background", 3),
    ("Vegetation", [90, 240, 0], "background", 4),
    ("Building", [140, 140, 140], "background", 5),
    ("Road", [100, 60, 100], "background", 6),
    ("GuardRail", [250, 100, 255], "background", 7),
    ("TrafficSign", [255, 255, 0], "background", 8),
    ("TrafficLight", [200, 200, 0], "background", 9),
    ("Pole", [255, 130, 0], "background", 10),
    ("Misc", [80, 80, 80], "background", 11),
    ("Truck", [160, 60, 60], "object", 12),
    ("Car", [255, 127, 80], "object", 13),
    ("Van", [0, 139, 139], "object", 14),
    ("Undefined", [0, 0, 0], "background", 0)
]

categories = [{
"id": 1,
"name": "Terrain",
"supercategory": "background"
}, {
"id": 2,
"name": "Sky",
"supercategory": "background"
}, {
"id": 3,
"name": "Tree",
"supercategory": "background"
}, {
"id": 4,
"name": "Vegetation",
"supercategory": "background"
}, {
"id": 5,
"name": "Building",
"supercategory": "background"
}, {
"id": 6,
"name": "Road",
"supercategory": "background"
}, {
"id": 7,
"name": "GuardRail",
"supercategory": "background"
}, {
"id": 8,
"name": "TrafficSign",
"supercategory": "background"
}, {
"id": 9,
"name": "TrafficLight",
"supercategory": "background"
},
{
"id": 10,
"name": "Pole",
"supercategory": "background"
},
{
"id": 11,
"name": "Misc",
"supercategory": "background"
},
{
"id": 12,
"name": "Truck",
"supercategory": "object"
},
{
"id": 13,
"name": "Car",
"supercategory": "object"
},
{
"id": 14,
"name": "Van",
"supercategory": "object"
},
{
"id": 15,
"name": "Undefined",
"supercategory": "background"
}]




def get_vkitti_files(dirName, ext):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)

    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_vkitti_files(fullPath, ext)
        elif fullPath.find("morning") != -1 and fullPath.find("Camera_0") != -1 and fullPath.find(ext) != -1:
            allFiles.append(fullPath)

    return allFiles

def get_img_obj(arg):
    im_id, img_filename = arg
    image = Image.open(img_filename)
    width, height = image.size
    file_name = img_filename.split("/")[-1]
    obj = {
        "id": im_id,
        "width": width,
        "height": height,
        "file_name": file_name,
        "license": None,
        "flickr_url": "",
        "coco_url": None,
        "date_captured": "",
        "loc": img_filename
    }

    return obj




def kitti_2_coco_crop(rgb_root, instance_seg_root, semantic_seg_root, ann_file):

    with open(ann_file) as f:
        ann_data = json.load(f)

    data = {"info": {
        "year": 2021,
        "date_created": "2020-11-13T07:52:01Z",
        "version": "1.0",
        "description": "VKITTI2",
        "contributor": "",
        "url": "https://app.hasty.ai/projects/e9305a22-1450-48ac-8b12-a601314caa23"
    }}

    data["licenses"] = []
    data["categories"] = categories
    data["annotations"] = []

    # images = get_vkitti_files(rgb_root, "jpg")
    # image_list = list(map(get_img_obj, list(enumerate(images))))

    data["images"] = ann_data["images"]

    # -----instance seg images----------
    instance_seg_files = get_vkitti_files(instance_seg_root, "png")
    semantic_seg_files = get_vkitti_files(semantic_seg_root, "png")
    # print(instance_seg_files)
    for idx, img in enumerate(data["images"]):
        
        img_filename = img["loc"]
        ann_img = list(filter(lambda ann_im: ann_im["loc"] == img_filename, ann_data["images"]))[0]
        ann_semantic_img_filename = ann_img["semseg_img_filename"]

        scene = img_filename.split("/")[-6]
        basename = img_filename.split(".")[-2].split("_")[-1]

        instance_img_filename = [s for s in instance_seg_files if (scene in s and basename in s)][0]
        semantic_img_filename = [s for s in semantic_seg_files if (scene in s and basename in s)][0]

        instance_img = Image.open(instance_img_filename)
        # crop instance image
        crop_t = transforms.CenterCrop(config_kitti.CROP_OUTPUT_SIZE)
        instance_img = crop_t(instance_img)

        instance_img_arr = np.asarray(instance_img)
        unique_values = np.unique(instance_img_arr)
        unique_values = np.delete(unique_values, np.where(unique_values == 0))
        
        semseg_img = Image.open(semantic_img_filename)
        semseg_img = crop_t(semseg_img)
        semseg_img_arr = np.asarray(semseg_img)

        data["images"][idx]["semseg_img_filename"] = ann_semantic_img_filename

        u_values = np.unique(semseg_img_arr)
        u_values = np.delete(u_values, np.where(u_values == 0))

        for instance in unique_values:

            coors = np.where(instance_img_arr==instance)

            
            coors_zip = list(zip(coors[0], coors[1]))

            # TODO: use ONE of those coordinates to find the class from the semantic seg image
            sample_coor = coors_zip[0]
            # print(sample_coor)
            rgb = semseg_img_arr[sample_coor]
            cat = list(filter(lambda rgb2class_tup: (rgb2class_tup[1] == rgb).all(), rgb_2_class))[0]
            cat_name = cat[0]
            # generate mask and convert it to coco using pycoco tools

            mask = np.zeros_like(instance_img_arr)
 
            mask[coors[0], coors[1]] = 1

            # plt.imshow(mask)
            # plt.show()

            encoded_mask = coco_mask.encode(np.asfortranarray(mask))
            bbx = coco_mask.toBbox(encoded_mask)
            area = coco_mask.area(encoded_mask)
            
            encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")

            category_id = list(filter(lambda category: category["name"] == cat_name, categories))[0]["id"]

            annotation_obj={
                "id": len(data["annotations"]),
                "image_id": img["id"],
                "category_id": category_id,
                "segmentation": encoded_mask,
                "area": int(area),
                "bbox": list(bbx),
                "iscrowd": 0
            }

            data["annotations"].append(annotation_obj)

            # break
        # break
    with open('kitti2coco_ann_crop.json', 'w') as outfile:
        json.dump(data, outfile)
        



            




imgs_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data/vkitti_2.0.3_rgb/")

instance_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data/vkitti_2.0.3_instanceSegmentation/")


semantic_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data/vkitti_2.0.3_classSegmentation/")


ann_file = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../kitti2coco_ann.json")

# semantic_map_dest = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_vkitti/virtual_world_vkitti-2/semseg_bin/")

# kitti_2_coco(imgs_root, instance_root, semantic_root, semantic_map_dest)
kitti_2_coco_crop(imgs_root, instance_root, semantic_root, ann_file)

# root = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_vkitti/virtual_world_vkitti-2/semseg_bin/")


