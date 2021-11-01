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
import models
import constants
import config_kitti

import sys
from pathlib import Path
import matplotlib.pyplot as plt

from argparse import ArgumentParser



torch.cuda.empty_cache()


all_categories, stuff_categories, thing_categories = get_stuff_thing_classes(config_kitti.COCO_ANN)

def save_fig(im, file_name, summary, dst):

    Path(dst).mkdir(parents=True, exist_ok=True)

    this_path = os.path.dirname(__file__)

    print(file_name, summary)
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

    ax.imshow(im,  interpolation='nearest', aspect='auto')
    plt.axis('off')
    fig.savefig(os.path.join(
        this_path, '../{}/{}.png'.format(dst, file_name)))
    plt.close(fig)


def inference_panoptic(model, data_loader_val, args):

    

    checkpoint = torch.load(args.checkpoint)
    
    model.load_state_dict(checkpoint['state_dict'])


    for images, _, _, _, _, _, _, _, basename in data_loader_val:

        imgs = list(img for img in images)
       
        # annotations = [{k: v.to(args.gpu) for k, v in t.items()}
        #                for t in anns]


        imgs = tensorize_batch(imgs, args.gpu)
        model.eval()

        with torch.no_grad():
            outputs = model(imgs)

            threshold_preds = threshold_instances(outputs)
            sorted_preds = sort_by_confidence(threshold_preds)


            if len(sorted_preds[0]["masks"]) > 0:

                # Get intermediate prediction and semantice prediction
                inter_pred_batch, sem_pred_batch, summary_batch = panoptic_fusion(
                    sorted_preds, all_categories, stuff_categories, thing_categories, threshold_by_confidence=False, sort_confidence=False)
                canvas = panoptic_canvas(
                    inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories)[0]

                # if canvas is None:
                #     return frame, summary_batch, sorted_preds
                # else:
                # TODO: generalize for a batch
                im = apply_panoptic_mask_gpu(
                    imgs[0], canvas).cpu().permute(1, 2, 0).numpy()
                
                #Save results
                dst = os.path.join(constants.INFERENCE_RESULTS, "Panoptic_seg")
                save_fig(im, basename, summary_batch[0], dst)
            
            else:
                print("No objects detected for: ", basename)



def inference(gpu, args):
    args.gpu = gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)

    model = models.get_model_by_name("PanopticSeg")
    
    model.to(device)

    imgs_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "..", config_kitti.DATA, "vkitti_2.0.3_rgb/")

    depth_root = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", config_kitti.DATA, "vkitti_2.0.3_depth/")

    annotation = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", config_kitti.COCO_ANN)


    _, data_loader_val = get_dataloaders(
        args.batch_size,
        imgs_root,
        depth_root,
        annotation,
        num_replicas=1,
        rank=0,
        split=True,
        val_size=config_kitti.VAL_SIZE,
        n_samples=config_kitti.MAX_TRAINING_SAMPLES,
        sampler=False,
        shuffle=False)

    inference_panoptic(model, data_loader_val, args)
           



