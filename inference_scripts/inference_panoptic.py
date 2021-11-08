# %%

import os
import os.path
import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from utils.tensorize_batch import tensorize_batch
from utils.panoptic_fusion import panoptic_fusion, panoptic_canvas, threshold_instances, sort_by_confidence, filter_by_class
from utils.get_vkitti_dataset_full import get_dataloaders
from utils.apply_panoptic_mask_gpu import apply_panoptic_mask_gpu
from utils.get_stuff_thing_classes import get_stuff_thing_classes
from utils.tracker import get_tracked_objects

import matplotlib.patches as patches

import models
import constants
import config_kitti

import sys
from pathlib import Path
import matplotlib.pyplot as plt

from argparse import ArgumentParser

iou_threshold = 0.2

torch.cuda.empty_cache()


all_categories, stuff_categories, thing_categories = get_stuff_thing_classes(config_kitti.COCO_ANN)

def map_cat(cats_arr, all_cat, things_cat):

    # map cat to names

    cat_names = list(map(lambda c: things_cat[c-1]["name"], cats_arr))

    # map name to obj
    objs = list(map(lambda name: list(filter(lambda obj: obj["name"] == name, all_cat))[0], cat_names))

    new_cats = list(map(lambda obj: obj["name"], objs))

    return new_cats


def save_fig(im, file_name, summary, dst, boxes, labels, ids, args):

    labels = map_cat(labels, args.all_categories, args.thing_categories)

    Path(dst).mkdir(parents=True, exist_ok=True)

    this_path = os.path.dirname(__file__)

    
    height, width = im.shape[:2]
    

    # file_name_basename = os.path.basename(file_name)
    # file_name = os.path.splitext(file_name_basename)[0]

    dppi = 96
    fig, ax = plt.subplots(1, 1, figsize=(
        width/dppi, height/dppi), dpi=dppi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    c = 1
    for obj in summary:
        ax.text(20, 30*c, '{}: {}'.format(obj["name"], obj["count_obj"]), style='italic',
                bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})
        c = c + 1

    
    for idx, box in enumerate(boxes):

        label = labels[idx]

        if len(ids)>0:
            obj_id = ids[idx]
        else:
            obj_id = "-"

        x1, y1, x2, y2 = box

        x_delta = x2 - x1
        y_delta = y2 - y1

        rect = patches.Rectangle((x1, y1), x_delta, y_delta, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(x1, y1 - 10 , "{}, id: {}".format(label, obj_id))
    

   
    ax.imshow(im,  interpolation='nearest', aspect='auto')
    # plt.axis('off')
    
    fig.savefig(os.path.join(
        this_path, '../{}/{}.png'.format(dst, file_name)))
    plt.close(fig)


def inference_panoptic(model, data_loader_val, args):

    

    checkpoint = torch.load(args.checkpoint)
    
    model.load_state_dict(checkpoint['state_dict'])

    prev_det = None
    new_det = None

    for images, basename in data_loader_val:

        imgs = list(img for img in images)
       
        # annotations = [{k: v.to(args.gpu) for k, v in t.items()}
        #                for t in anns]


        imgs = tensorize_batch(imgs, args.gpu)
        model.eval()

        with torch.no_grad():
            outputs = model(imgs)

            threshold_preds = threshold_instances(outputs)
            sorted_preds = sort_by_confidence(threshold_preds)
            ids = []

        if config_kitti.OBJECT_TRACKING:
            tracked_obj = None
            if prev_det is None:
                tracked_obj = get_tracked_objects(
                    None, sorted_preds[0]["boxes"], None, sorted_preds[0]["labels"], iou_threshold, args.gpu)
            else:
                tracked_obj = get_tracked_objects(
                    prev_det[0]["boxes"], sorted_preds[0]["boxes"], prev_det[0]["labels"], sorted_preds[0]["labels"], iou_threshold, args.gpu)

            sorted_preds[0]["ids"] = tracked_obj
            ids = sorted_preds[0]["ids"]
            if len(tracked_obj) > 0:
                sorted_preds[0]["boxes"] = sorted_preds[0]["boxes"][:len(
                    tracked_obj)]
                sorted_preds[0]["masks"] = sorted_preds[0]["masks"][:len(
                    tracked_obj)]
                sorted_preds[0]["scores"] = sorted_preds[0]["scores"][:len(
                    tracked_obj)]
                sorted_preds[0]["labels"] = sorted_preds[0]["labels"][:len(
                    tracked_obj)]

        if len(sorted_preds[0]["masks"]) > 0:

            # Get intermediate prediction and semantice prediction
            inter_pred_batch, sem_pred_batch, summary_batch = panoptic_fusion(
                sorted_preds, all_categories, stuff_categories, thing_categories, args.gpu, threshold_by_confidence=False, sort_confidence=False)
            canvas = panoptic_canvas(
                inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories, args.gpu)[0]

            # if canvas is None:
            #     return frame, summary_batch, sorted_preds
            # else:
            # TODO: generalize for a batch
            im = apply_panoptic_mask_gpu(
                imgs[0], canvas).cpu().permute(1, 2, 0).numpy()
            
            #Save results
            boxes =sorted_preds[0]["boxes"]
            labels =sorted_preds[0]["labels"]

            dst = os.path.join(args.dst)
            save_fig(im, basename[0], summary_batch[0], dst, boxes, labels, ids, args)

            prev_det = sorted_preds
        
        else:

            semantic_logits = sorted_preds[0]["semantic_logits"]
            semantic_mask = torch.argmax(semantic_logits, dim=0)
            im = apply_panoptic_mask_gpu(
                imgs[0], semantic_mask).cpu().permute(1, 2, 0).numpy()
            
            dst = os.path.join(args.dst)
            save_fig(im, basename[0], [], dst, [], [], [], args)
            print("No objects detected for: ", basename[0])



def inference(gpu, args):
    args.gpu = gpu
    
    print("DEVICE", args.gpu)
    model = models.get_model_by_name("PanopticSeg")
    
    model.to(args.gpu)

    imgs_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "..", args.data_folder, "vkitti_2.0.3_rgb/")


    annotation = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", args.categories_json)

    all_categories, _, thing_categories = get_stuff_thing_classes(annotation)

    args.all_categories = all_categories
    args.thing_categories = thing_categories

    if args.dataloader is None:

        if args.data_folder is None:
            raise ValueError("Either datalodar or data_folder has to be provided")


        data_loader, _ = get_dataloaders(
        args.batch_size,
        imgs_root,
        None,
        None,
        num_replicas=1,
        rank=0,
        split=False,
        val_size=None,
        n_samples=config_kitti.MAX_TRAINING_SAMPLES,
        sampler=False,
        shuffle=False,
        is_test_set=True)
    else:
        data_loader = args.dataloader

    

    inference_panoptic(model, data_loader, args)
           



