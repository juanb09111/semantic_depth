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



device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

temp_variables.DEVICE = device
torch.cuda.empty_cache()



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




def eval_sem_seg(model, data_loader_val, weights_file):

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # load weights
    print("eval depth completion weights: ", weights_file)
    model.load_state_dict(torch.load(weights_file))
    # move model to the right device
    model.to(device)

    iou_arr = []
    for images, anns, _, _, _, _, _, _, _ in data_loader_val:

        imgs = list(img for img in images)
       
        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in anns]

        imgs = tensorize_batch(imgs, device)
        model.eval()

        with torch.no_grad():
            outputs = model(imgs, anns=annotations)

            for idx, out in enumerate(outputs):


                #Calculate miou
                semantic_mask = anns[idx]["semantic_mask"]
                semantic_logits = out["semantic_logits"]
                iou = mIOU(semantic_mask, semantic_logits)
                iou_arr.append(iou)

        torch.cuda.empty_cache()
    return np.mean(iou_arr), imgs[0], anns[0]["semantic_logits"], outputs[0]["semantic_logits"]

