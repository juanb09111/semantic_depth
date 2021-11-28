# %%

import os
import os.path
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torchvision.utils import save_image

from utils.tensorize_batch import tensorize_batch
from utils.panoptic_fusion import panoptic_fusion, panoptic_canvas, threshold_instances, sort_by_confidence, filter_by_class
from utils.get_vkitti_dataset_full import get_dataloaders
from utils.apply_panoptic_mask_gpu import apply_panoptic_mask_gpu
from utils.get_stuff_thing_classes import get_stuff_thing_classes
from utils.tracker import get_tracked_objects

import matplotlib.patches as patches
from PIL import Image as im

import models
import constants
import config_kitti

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import json

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


def save_mask(mask, file_name, dst, args):
    
    this_path = os.path.dirname(__file__)
    
    dst_folder = os.path.join(
        this_path, '../{}/{}'.format(constants.INFERENCE_RESULTS, dst))
    
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    dst = os.path.join(dst_folder, '{}.png'.format(file_name))

    mask_numpy = mask.cpu().numpy()
    # height, width = mask_numpy.shape[:2]
    data = im.fromarray(mask_numpy.astype(np.uint8))
    
    data.save(dst)
 

def save_fig(im, file_name, summary, dst, boxes, labels, ids, args):
    
    this_path = os.path.dirname(__file__)
    
    dst_folder = os.path.join(
        this_path, '../{}/{}'.format(constants.INFERENCE_RESULTS, dst))
    
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    dst = os.path.join(dst_folder, '{}.png'.format(file_name))

    labels = map_cat(labels, args.all_categories, args.thing_categories)

    
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
    
    fig.savefig(dst, format="png")
    plt.close(fig)

def get_ann_obj(canvas, ids_label_map, all_categories, thing_categories):
    # print(ids_label_map)
    obj_arr = []
    unique_val, counts = torch.unique(canvas, return_counts=True)
    # print(unique_val, counts, ids_label_map)
    for idx, val in enumerate(unique_val):

        if val != 0:
            id_2_label = list(filter(lambda a: a[0] == val, ids_label_map))[0]

            if id_2_label[2]["isthing"]:
                # print(id_2_label)
                category_idx = id_2_label[1]
                # print(category_id, thing_categories)
                # cat = list(filter(lambda a: a["id"] == category_id, thing_categories))[0]
                cat = thing_categories[category_idx - 1]

                cat_id = list(filter(lambda a: a["name"] == cat["name"], all_categories))[0]["id"]
            else:
                category_id = id_2_label[1]
                cat = list(filter(lambda a: a["id"] == category_id, all_categories))[0]
                cat_id = id_2_label[1].item()

            obj = {
                "id": val.item(),
                "area": counts[idx].item(),
                "isthing": id_2_label[2]["isthing"],
                "category_id": cat_id,
                "cat_name": cat["name"]
            }
            # print(obj)
            obj_arr.append(obj)
    
    return obj_arr


def inference_panoptic(model, data_loader_val, args):

    with open(args.annotation) as coco_file:
        # read file
        data = json.load(coco_file)


    res_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": data["categories"],
        "annotations":[]
    }

    checkpoint = torch.load(args.checkpoint)
    
    model.load_state_dict(checkpoint['state_dict'])

    prev_det = None
    new_det = None

    id = 1

    # for images, _, _, _, _, _, _, _, basename in data_loader_val:
    for images, basename in data_loader_val:
        
        
        # segments_info = []
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
            inter_pred_batch, sem_pred_batch, summary_batch, ids_label_map_batch = panoptic_fusion(
                sorted_preds, all_categories, stuff_categories, thing_categories, args.gpu, threshold_by_confidence=False, sort_confidence=False)

            canvas = panoptic_canvas(
                inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories, args.gpu)[0]

            # unique_val = torch.unique(canvas)
            # print(basename[0], unique_val, ids_label_map_batch[0])
            # print("basename", basename[0])
            obj_arr = get_ann_obj(canvas, ids_label_map_batch[0], args.all_categories, args.thing_categories)

            res_data["annotations"].append({"segments_info": obj_arr})
            image = {"file_name": basename[0]+".jpg", "id": id}
            res_data["images"].append(image)
            id = id+1
            # print(basename[0], summary_batch[0], obj_arr)
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
            save_mask(canvas,basename[0], dst, args)
            # save_fig(im, basename[0], [], dst, [], labels, ids, args)

            prev_det = sorted_preds
        
        else:

            semantic_logits = sorted_preds[0]["semantic_logits"]
            semantic_mask = torch.argmax(semantic_logits, dim=0)
            im = apply_panoptic_mask_gpu(
                imgs[0], semantic_mask).cpu().permute(1, 2, 0).numpy()
            
            dst = os.path.join(args.dst)
            save_fig(im, basename[0], [], dst, [], [], [], args)
            print("No objects detected for: ", basename[0])
        
        
        
        
    

    this_path = os.path.dirname(__file__)
    
    dst_folder = os.path.join(
        this_path, '../{}/{}'.format(constants.INFERENCE_RESULTS, args.dst))

    out_file = os.path.join(dst_folder, "pred.json")
    
    with open(out_file, 'w') as outfile:
        json.dump(res_data, outfile)



def inference(gpu, args):
    args.gpu = gpu
    rank = args.local_ranks * args.ngpus + gpu
    print("DEVICE", args.gpu)
    model = models.get_model_by_name("PanopticSeg")
    
    model.to(args.gpu)

    imgs_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "..", args.data_folder, "vkitti_2.0.3_rgb/")

    # semantic_root = os.path.join(os.path.dirname(os.path.abspath(
    #         __file__)), "..", config_kitti.DATA, "vkitti_2.0.3_classSegmentation/") 
        

    annotation = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", args.categories_json)

    args.annotation = annotation 

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
        annotation,
        num_replicas=args.world_size,
        rank=rank,
        split=False,
        val_size=None,
        n_samples=config_kitti.MAX_TRAINING_SAMPLES,
        sampler=False,
        shuffle=False,
        is_test_set=True)

    else:
        data_loader = args.dataloader

    

    inference_panoptic(model, data_loader, args)
           



