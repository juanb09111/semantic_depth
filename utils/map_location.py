
import os.path

from PIL import Image

import config_kitti

import matplotlib.pyplot as plt

import json


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


# def get_img_obj(arg):
#     im_id, img_filename = arg
#     loc = img_filename.split("/").index(config_kitti.RGB_DATA)
#     loc = "/".join(img_filename.split("/")[loc:])
#     image = Image.open(img_filename)
#     width, height = image.size
#     file_name = img_filename.split("/")[-1]
#     obj = {
#         "id": im_id,
#         "width": width,
#         "height": height,
#         "file_name": file_name,
#         "license": None,
#         "flickr_url": "",
#         "coco_url": None,
#         "date_captured": "",
#         "loc": loc
#     }

#     return obj

def get_img_obj(img_obj, file_list):
    
    file_name = img_obj["file_name"]
    full_path = list(filter(lambda fname: fname.split("/")[-1] == file_name, file_list))[0]
    loc = full_path.split("/").index(config_kitti.RGB_DATA)
    loc = "/".join(full_path.split("/")[loc:])
    obj = {**img_obj, "loc": loc}

    return obj




def map_loc(rgb_root,ann_file ):
    with open(ann_file) as f:
        ann_data = json.load(f)


    images = sorted(get_vkitti_files(rgb_root, "jpg"))
    
    # image_list = list(map(get_img_obj, list(enumerate(images))))

    img_list = []
    for img_obj in ann_data["images"]:

        new_obj = get_img_obj(img_obj, images)
        img_list.append(new_obj)

    ann_data["images"] = img_list

    with open("COCO_ann_loc_03_2022.json", 'w') as outfile:
        json.dump(ann_data, outfile)


# data_folder_name = config_kitti.DATA
data_folder_name = "data_jd_2022"

imgs_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "..",  data_folder_name, config_kitti.RGB_DATA)

ann_file = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "..", config_kitti.COCO_ANN)

map_loc(imgs_root, ann_file)

