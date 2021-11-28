# %%

import os
import os.path
import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from utils.tensorize_batch import tensorize_batch
from utils.convert_tensor_to_RGB import convert_tensor_to_RGB
from utils.get_vkitti_dataset_full import get_dataloaders
import models
import constants
import config_kitti

import sys
from pathlib import Path
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import models


torch.cuda.empty_cache()


def save_fig(im, loc, file_name, shape):

    height, width = shape
    # im = im.cpu().permute(1, 2, 0).numpy()

    dppi = 96
    fig, ax = plt.subplots(1, 1, figsize=(
        width/dppi, height/dppi), dpi=dppi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ax.imshow(im,  interpolation='nearest', aspect='auto')
    plt.axis('off')
    fig.savefig(os.path.join('{}/{}.png'.format(loc, file_name)))
    plt.close(fig)


def inference_sem_seg_depth(model, data_loader_val, weights_file):

    # load weights
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[0], find_unused_parameters=True
    )

    dist.barrier()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}

    checkpoint = torch.load(args.checkpoint, map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])

    # move model to the right device

    for images, anns, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, sparse_depth_gt_full, basename in data_loader_val:

        imgs = list(img for img in images)
        lidar_fov = list(lid_fov for lid_fov in lidar_fov)
        masks = list(mask for mask in masks)
        sparse_depth = list(sd for sd in sparse_depth)
        k_nn_indices = list(k_nn for k_nn in k_nn_indices)
        sparse_depth_gt = list(sp_d for sp_d in sparse_depth_gt)
        sparse_depth_gt_full = list(
            sp_d_full for sp_d_full in sparse_depth_gt_full)

        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in anns]

        imgs = tensorize_batch(imgs, device)
        lidar_fov = tensorize_batch(lidar_fov, device, dtype=torch.float)
        masks = tensorize_batch(masks, device, dtype=torch.bool)
        sparse_depth = tensorize_batch(sparse_depth, device)
        k_nn_indices = tensorize_batch(
            k_nn_indices, device, dtype=torch.long)
        sparse_depth_gt = tensorize_batch(
            sparse_depth_gt, device, dtype=torch.float)  # BxCxHW
        sparse_depth_gt_full = tensorize_batch(
            sparse_depth_gt_full, device, dtype=torch.float)

        semantic_masks = list(
            map(lambda ann: ann['semantic_mask'], annotations))

        semantic_masks = tensorize_batch(semantic_masks, device)

        model.eval()

        with torch.no_grad():
            outputs = model(imgs,  sparse_depth,
                            masks,
                            lidar_fov,
                            k_nn_indices,
                            sparse_depth_gt=None,
                            semantic_masks=None)

            for idx, out in enumerate(outputs):
                out_depth = outputs[idx]["depth"]
                shape = out_depth.shape

                out_depth_numpy = out_depth.cpu().numpy()/255
                # --------------------------------------

                # Calculate miou
                semantic_mask = semantic_masks[idx]
                semantic_logits = out["semantic_logits"]

                mask_output = torch.argmax(semantic_logits, dim=0)
                mask_output = convert_tensor_to_RGB(
                    mask_output.unsqueeze(0), device).squeeze(0)/255
                mask_output = mask_output.cpu().permute(1, 2, 0).numpy()

                rgb = imgs[idx].cpu().permute(1, 2, 0).numpy()

                loc = "{}/{}".format(constants.INFERENCE_RESULTS,
                                     "Semseg_Depth_v4")
                # Create folde if it doesn't exist
                Path(loc).mkdir(parents=True, exist_ok=True)

                filename_rgb = basename[idx] + "_rgb"
                filename_semantic = basename[idx] + "_semantic"
                filename_depth = basename[idx] + "_depth"

                save_fig(rgb, loc, filename_rgb, shape)
                save_fig(out_depth_numpy, loc, filename_depth, shape)
                save_fig(mask_output, loc, filename_semantic, shape)
        # torch.cuda.empty_cache()
    # return np.mean(rmse_arr), np.mean(iou_arr), imgs[0], anns[0]["semantic_mask"], outputs[0]["semantic_logits"], sparse_depth_gt[0], sparse_depth_gt_full[0].squeeze_(0), outputs[0]["depth"]


def inference(gpu, args):
    args.gpu = gpu
    rank = args.local_ranks * args.ngpus + gpu

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=0
    )
    torch.manual_seed(0)

    print("DEVICE", args.gpu)
    model = models.get_model_by_name("Semseg_Depth_v4")

    model.to(args.gpu)

    imgs_root = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", args.data_folder, "vkitti_2.0.3_rgb/")

    depth_root = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), args.data_folder, "vkitti_2.0.3_depth/")

    semantic_root = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", args.data_folder, "semseg_bin/")

    annotation = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", args.categories_json)

    args.annotation = annotation

    if args.dataloader is None:

        if args.data_folder is None:
            raise ValueError(
                "Either datalodar or data_folder has to be provided")

        data_loader, _ = get_dataloaders(
            args.batch_size,
            imgs_root,
            semantic_root,
            depth_root,
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

    inference_sem_seg_depth(model, data_loader, args.checkpoint)


