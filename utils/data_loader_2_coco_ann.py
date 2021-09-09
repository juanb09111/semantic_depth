import torch
import json
import constants
import config_kitti
import os.path

def data_loader_2_coco_ann(data_loader_val_filename, coco_ann_json):
    
    data_loader_val = torch.load(data_loader_val_filename) 
    with open(coco_ann_json) as coco_file:
        # read file
        data_all = json.load(coco_file)

    data_val = {
        "info": data_all["info"],
        "licenses": data_all["licenses"],
        "categories": data_all["categories"],
        "images": [],
        "annotations": []
    }
    for _, anns, _, _, _, _, _, _, _ in data_loader_val:


        for idx, _ in enumerate(anns):

            image_id = anns[idx]['image_id'].cpu().data

            image_obj = list(filter(lambda im: im["id"] == image_id, data_all["images"]))[0]

            data_val["images"].append(image_obj)

            annotations = list(filter(lambda an: an["image_id"] == image_id, data_all["annotations"]))

            data_val["annotations"] = [*data_val["annotations"], *annotations]

    val_ann_filename = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../", constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME)

    with open(val_ann_filename, 'w') as outfile:
        json.dump(data_val, outfile)



# data_loader_val_filename = os.path.join(os.path.dirname(
#         os.path.abspath(__file__)), "../", constants.DATA_LOADERS_LOC, "vkitti_data_loader_val_100_obj.pth")

# coco_ann_json = os.path.join(os.path.dirname(
#         os.path.abspath(__file__)), "../", config_kitti.COCO_ANN)


# data_loader_2_coco_ann(data_loader_val_filename, coco_ann_json)