# %%
import os
import os.path
import numpy as np
import copy
import torch

from utils.tensorize_batch import tensorize_batch
from utils.panoptic_fusion import (
    threshold_instances,
    sort_by_confidence
)
from utils.get_youtube_dataloader import get_dataloaders
from utils.apply_instance_mask import apply_instance_masks
from utils.get_stuff_thing_classes import get_stuff_thing_classes
from utils.tracker_recycle import get_tracked_objects

import matplotlib.patches as patches
from PIL import Image as im

import models
import constants
import config

from pathlib import Path
import matplotlib.pyplot as plt
import json


iou_threshold = 0.2

torch.cuda.empty_cache()


# all_categories, stuff_categories, thing_categories = get_stuff_thing_classes(config.COCO_ANN)


def map_cat(cats_arr, all_cat, things_cat):

    # map cat to names

    cat_names = list(map(lambda c: things_cat[c - 1]["name"], cats_arr))

    # map name to obj
    objs = list(
        map(
            lambda name: list(filter(lambda obj: obj["name"] == name, all_cat))[0],
            cat_names,
        )
    )

    new_cats = list(map(lambda obj: obj["name"], objs))

    return new_cats


def save_mask(mask, file_name, dst, args):

    this_path = os.path.dirname(__file__)

    dst_folder = os.path.join(
        this_path, "../{}/{}".format(constants.INFERENCE_RESULTS, dst)
    )

    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    dst = os.path.join(dst_folder, "{}.png".format(file_name))

    mask_numpy = mask.cpu().numpy()
    # height, width = mask_numpy.shape[:2]
    data = im.fromarray(mask_numpy.astype(np.uint8))

    data.save(dst)


def save_fig(im, file_name, dst, boxes, labels, ids,  unmatched_boxes, unmatched_labels, unmatched_ids, args, draw_boxes=True):

    this_path = os.path.dirname(__file__)

    dst_folder = os.path.join(
        this_path, "../{}/{}_vis".format(constants.INFERENCE_RESULTS, dst)
    )

    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    dst = os.path.join(dst_folder, "{}.png".format(file_name))

    labels = map_cat(labels, args.all_categories, args.thing_categories)
    unmatched_labels = map_cat(unmatched_labels, args.all_categories, args.thing_categories)

    height, width = im.shape[:2]

    # file_name_basename = os.path.basename(file_name)
    # file_name = os.path.splitext(file_name_basename)[0]

    dppi = 96
    fig, ax = plt.subplots(1, 1, figsize=(width / dppi, height / dppi), dpi=dppi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    

    if draw_boxes:
        for idx, box in enumerate(boxes):

            label = labels[idx]

            if len(ids) > 0:
                obj_id = ids[idx]
            else:
                obj_id = "-"

            x1, y1, x2, y2 = box.cpu().numpy()

            x_delta = x2 - x1
            y_delta = y2 - y1

            rect = patches.Rectangle(
                (x1, y1), x_delta, y_delta, linewidth=1, edgecolor="r", facecolor="none"
            )

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(
                x2,
                y2 - 10,
                "{}, id: {}".format(label, obj_id),
                color="white",
                fontsize=15,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
            )

        for idx, box in enumerate(unmatched_boxes):

            label = unmatched_labels[idx]

            if len(unmatched_ids) > 0:
                obj_id = unmatched_ids[idx]
            else:
                obj_id = "-"

            x1, y1, x2, y2 = box.cpu().numpy()

            x_delta = x2 - x1
            y_delta = y2 - y1

            rect = patches.Rectangle(
                (x1, y1), x_delta, y_delta, linewidth=1, edgecolor="yellow", facecolor="none"
            )

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(
                x2,
                y2 - 10,
                "{}, id: {}".format(label, obj_id),
                color="white",
                fontsize=15,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
            )

    ax.imshow(im, interpolation="nearest", aspect="auto")
    # plt.axis('off')

    fig.savefig(dst, format="png")
    plt.close(fig)




def inference_panoptic(model, data_loader_val, args):

    with open(args.annotation) as coco_file:
        # read file
        data = json.load(coco_file)

    res_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": data["categories"],
        "annotations": [],
    }

    checkpoint = torch.load(args.checkpoint)

    model.load_state_dict(checkpoint["state_dict"])

    prev_det = None
    new_det = None

    prev_img_filename = None
    new_img_filename = None

    id = 1

    # for images, _, _, _, _, _, _, _, basename in data_loader_val:
    for images, basename, img_filename in data_loader_val:

        # print(basename[0])
        new_img_filename = img_filename[0]
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

        if config.OBJECT_TRACKING:
            tracked_obj = None
            if prev_det is None:
                tracked_obj, untracked_obj = get_tracked_objects(
                    None,
                    new_img_filename,
                    None,
                    sorted_preds[0]["masks"],
                    None,
                    sorted_preds[0]["boxes"],
                    None,
                    sorted_preds[0]["labels"],
                    None,
                    sorted_preds[0]["scores"],
                    args.super_cat_indices,
                    iou_threshold,
                    args.algorithm,
                    args.gpu,
                )
            else:
                tracked_obj, untracked_obj = get_tracked_objects(
                    prev_img_filename,
                    new_img_filename,
                    prev_det[0]["masks"],
                    sorted_preds[0]["masks"],
                    prev_det[0]["boxes"],
                    sorted_preds[0]["boxes"],
                    prev_det[0]["labels"],
                    sorted_preds[0]["labels"],
                    prev_det[0]["scores"],
                    sorted_preds[0]["scores"],
                    args.super_cat_indices,
                    iou_threshold,
                    args.algorithm,
                    args.gpu,
                )

            sorted_preds[0]["ids"] = tracked_obj
            ids = sorted_preds[0]["ids"]
            
            if len(tracked_obj) > 0:
                sorted_preds[0]["boxes"] = sorted_preds[0]["boxes"][: len(tracked_obj)]
                sorted_preds[0]["masks"] = sorted_preds[0]["masks"][: len(tracked_obj)]
                sorted_preds[0]["scores"] = sorted_preds[0]["scores"][: len(tracked_obj)]
                sorted_preds[0]["labels"] = sorted_preds[0]["labels"][: len(tracked_obj)]
            
            #Extend sorted_preds with unmatched
            unmatched_boxes = []
            unmatched_labels = []
            unmatched_ids = []
            
            sorted_preds_extended = copy.deepcopy(sorted_preds)
            if untracked_obj is not None:
                unmatched_boxes= untracked_obj["boxes"]
                unmatched_masks = untracked_obj["masks"]
                unmatched_scores = untracked_obj["scores"]
                unmatched_labels= untracked_obj["labels"]
                unmatched_ids= untracked_obj["ids"]

                sorted_preds_extended[0]["boxes"] = torch.cat((sorted_preds[0]["boxes"], unmatched_boxes), 0)
                sorted_preds_extended[0]["masks"] = torch.cat((sorted_preds[0]["masks"], unmatched_masks), 0)
                sorted_preds_extended[0]["scores"] = torch.cat((sorted_preds[0]["scores"], unmatched_scores), 0)
                sorted_preds_extended[0]["labels"] = torch.cat((sorted_preds[0]["labels"], unmatched_labels), 0)
                sorted_preds_extended[0]["ids"] = torch.cat((sorted_preds[0]["ids"], unmatched_ids), 0)

        if len(sorted_preds[0]["masks"]) > 0:

            #Update prev_det
            prev_det = sorted_preds

            

            # unique_val = torch.unique(canvas)
            # print(basename[0], unique_val, ids_label_map_batch[0])
            # print("basename", basename[0])

            boxes = sorted_preds[0]["boxes"]
            labels = sorted_preds[0]["labels"]
            

            
            # print("ids:",  len(ids)+ len(unmatched_ids))
            # im = apply_panoptic_mask_gpu(imgs[0], canvas).cpu().permute(1, 2, 0).numpy()
            im = apply_instance_masks(imgs[0], prev_det[0]["masks"], 0.5, args.gpu, ids=prev_det[0]["ids"])
            # Save results

            dst = os.path.join(args.dst)


            # Visualize results
            save_fig(im, basename[0], dst, boxes, labels, ids, unmatched_boxes, unmatched_labels, unmatched_ids, args, draw_boxes=True)

            prev_img_filename = img_filename[0]

        else:

            print("No objects detected for: ", basename[0])




def inference(gpu, args):
    args.gpu = gpu
    rank = args.local_ranks * args.ngpus + gpu
    print("DEVICE", args.gpu)
    model = models.get_model_by_name("MaskRcnn")

    model.to(args.gpu)

    imgs_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        args.data_folder
    )


    annotation = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", args.categories_json
    )

    args.annotation = annotation

    all_categories, stuff_categories, thing_categories = get_stuff_thing_classes(
        annotation
    )
    things_names = list(map(lambda thing: thing["name"], thing_categories))
    super_cat_indices = [
        things_names.index(label) + 1 for label in config.SUPER_CAT
    ]

    args.all_categories = all_categories
    args.thing_categories = thing_categories
    args.stuff_categories = stuff_categories
    args.super_cat_indices = super_cat_indices

    if args.dataloader is None:

        if args.data_folder is None:
            raise ValueError("Either datalodar or data_folder has to be provided")

        data_loader_test = get_dataloaders(
            args.batch_size,
            imgs_root,
            None,
            num_replicas=args.world_size,
            rank=rank,
            split=False,
            val_size=None,
            n_samples=None,
            sampler=False,
            shuffle=False,
            is_test_set=True)

    else:
        data_loader_test = args.dataloader

    inference_panoptic(model, data_loader_test, args)

