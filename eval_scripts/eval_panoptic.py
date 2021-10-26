# %%
"""This module performs AP evaluation using coco_eval"""

import json
import os.path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.tensorize_batch import tensorize_batch
from pycocotools import mask, cocoeval
from pycocotools.coco import COCO
import models
import constants
import config_kitti
import temp_variables
import sys
import matplotlib.pyplot as plt




torch.cuda.empty_cache()


def map_cat(cats_arr, all_cat, things_cat):

    # map cat to names

    cat_names = list(map(lambda c: things_cat[c-1]["name"], cats_arr))

    # map name to obj
    objs = list(map(lambda name: list(filter(lambda obj: obj["name"] == name, all_cat))[0], cat_names))

    new_cats = list(map(lambda obj: obj["id"], objs))

    return new_cats


def mIOU(label, pred):
    with torch.no_grad():
        # Include background
        num_classes = config_kitti.NUM_STUFF_CLASSES + config_kitti.NUM_THING_CLASSES + 1
        pred = F.softmax(pred, dim=0)
        pred = torch.argmax(pred, dim=0).squeeze(1)
        iou_list = list()
        present_iou_list = list()

        pred = pred.view(-1)
        label = label.view(-1)
        # Note: Following for loop goes from 0 to (num_classes-1)
        # and ignore_index is num_classes, thus ignore_index is
        # not considered in computation of IoU.
        for sem_class in range(1, num_classes):
            pred_inds = (pred == sem_class)
            target_inds = (label == sem_class)
            if target_inds.long().sum().item() == 0:
                iou_now = float('nan')
            else:
                intersection_now = (pred_inds[target_inds]).long().sum().item()
                union_now = pred_inds.long().sum().item() + \
                    target_inds.long().sum().item() - intersection_now
                iou_now = float(intersection_now) / float(union_now)
                present_iou_list.append(iou_now)
            iou_list.append(iou_now)
    return np.mean(present_iou_list)




def eval_panoptic(model, data_loader_val, weights_file, all_cat, things_cat, device):

   
    # load weights
    print("eval depth completion weights: ", weights_file)
    model.load_state_dict(torch.load(weights_file)["state_dict"])
    # move model to the right device
    model.to(device)
    res = []
    iou_arr = []
    for images, anns, _, _, _, _, _, _, _ in data_loader_val:

        imgs = list(img for img in images)
       
        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in anns]

        semantic_masks = list(
            map(lambda ann: ann['semantic_mask'], annotations))

        semantic_masks = tensorize_batch(semantic_masks, device)

       
        imgs = tensorize_batch(imgs, device)
        model.eval()

        with torch.no_grad():
            outputs = model(imgs, anns=annotations)

            for idx, out in enumerate(outputs):


                #Calculate miou
                semantic_mask = semantic_masks[idx]
                semantic_logits = out["semantic_logits"]
                iou = mIOU(semantic_mask, semantic_logits)
                iou_arr.append(iou)


                #COCO eval
                image_id = anns[idx]['image_id'].cpu().data
                pred_scores = out["scores"].cpu().data.numpy()
                pred_masks = []
                pred_boxes = []
                pred_labels = out['labels'].cpu().data.numpy()
                # print("labels", pred_labels)
                if "masks" in out.keys():
                    pred_masks = out["masks"].cpu().data.numpy()

                if "boxes" in out.keys():
                    pred_boxes = out["boxes"].cpu().data.numpy()
                    # print("len boxes:", len(pred_boxes))

                mapped_pred_labels = map_cat(
                    np.array(pred_labels), all_cat, things_cat)

                for i, _ in enumerate(pred_scores):
                    if int(pred_labels[i]) > 0:
                        obj = {"image_id": image_id[0].numpy().tolist(),
                               "category_id": mapped_pred_labels[i],
                               "score": pred_scores[i].item()}
                        if "masks" in out.keys():
                            bimask = pred_masks[i] > 0.5
                            bimask = np.array(
                                bimask[0, :, :, np.newaxis], dtype=np.uint8, order="F")

                            encoded_mask = mask.encode(
                                np.asfortranarray(bimask))[0]
                            encoded_mask['counts'] = encoded_mask['counts'].decode(
                                "utf-8")
                            obj['segmentation'] = encoded_mask
                        if "boxes" in out.keys():
                            bbox = pred_boxes[i]
                            bbox_coco = [int(bbox[0]), int(bbox[1]), int(
                                bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])]
                            obj['bbox'] = bbox_coco

                        res.append(obj)

        torch.cuda.empty_cache()
    
    # COCO evaluation 
    val_ann_filename = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME)

    # COCO res file
    with open(constants.COCO_RES_JSON_FILENAME, 'w') as res_file:
        json.dump(res, res_file)

    # # Make coco api from annotation file
    coco_gt = COCO(val_ann_filename)


    # Load res with coco.loadRes
    coco_dt = coco_gt.loadRes(constants.COCO_RES_JSON_FILENAME)
    # Get the list of images
    img_ids = sorted(coco_gt.getImgIds())

    train_res_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", constants.RES_LOC, constants.EVAL_RES_FILENAME)
    
    fo = open(train_res_file, 'a+')
    sys.stdout = fo
    for iou_type in config_kitti.IOU_TYPES:
        coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.img_ids = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    

    return np.mean(iou_arr), imgs[0], anns[0]["semantic_mask"], outputs[0]["semantic_logits"]

