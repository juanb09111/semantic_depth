# %%
"""This module performs AP evaluation using coco_eval"""

import json
import os.path
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from utils.tensorize_batch import tensorize_batch
from utils.get_vkitti_dataset_full import get_dataloaders


import models
import constants
import config_kitti

import sys
import matplotlib.pyplot as plt


from argparse import ArgumentParser



torch.cuda.empty_cache()


def RMSE(sparse_depth_gt, pred, device):
    with torch.no_grad():
        mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=device,
                                                                dtype=torch.float64), torch.tensor((0), device=device, dtype=torch.float64))
        mask_gt = mask_gt.squeeze_(1)

        sparse_depth_gt = sparse_depth_gt.squeeze_(
            0)  # remove C dimension there's only one

        c = torch.tensor((1000), device=device)
        sparse_depth_gt = sparse_depth_gt*c
        pred = pred*c

        criterion = nn.MSELoss()

        res = torch.sqrt(criterion(sparse_depth_gt*mask_gt, pred*mask_gt))

    return res


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


def eval_sem_seg_depth(model, data_loader_val, weights_file, device):

    # load weights
    print("eval depth completion weights: ", weights_file)
    model.load_state_dict(torch.load(weights_file)["state_dict"])
    # move model to the right device
    model.to(device)

    rmse_arr = []
    iou_arr = []
    for images, anns, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, sparse_depth_gt_full, _ in data_loader_val:

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

                # --------------------------------------
                rmse = RMSE(sparse_depth_gt[idx], out_depth, device)
                rmse_arr.append(rmse.cpu().data.numpy())

                # Calculate miou
                semantic_mask = semantic_masks[idx]
                semantic_logits = out["semantic_logits"]
                iou = mIOU(semantic_mask, semantic_logits)
                iou_arr.append(iou)

        torch.cuda.empty_cache()
    return np.mean(rmse_arr), np.mean(iou_arr), imgs[0], anns[0]["semantic_mask"], outputs[0]["semantic_logits"], sparse_depth_gt[0], sparse_depth_gt_full[0].squeeze_(0), outputs[0]["depth"]


def eval_set(model, data_loader_val, weights_file):

     # load weights
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[0], find_unused_parameters=True
    )

    dist.barrier()
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}

    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    
    model.load_state_dict(checkpoint['state_dict'])


    rmse_arr = []
    iou_arr = []
    for images, anns, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, sparse_depth_gt_full, _ in data_loader_val:

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

                # --------------------------------------
                rmse = RMSE(sparse_depth_gt[idx], out_depth, device)
                rmse_arr.append(rmse.cpu().data.numpy())

                # Calculate miou
                semantic_mask = semantic_masks[idx]
                semantic_logits = out["semantic_logits"]
                iou = mIOU(semantic_mask, semantic_logits)
                iou_arr.append(iou)

        torch.cuda.empty_cache()
    return np.mean(rmse_arr), np.mean(iou_arr)
    


if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)

    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--ip_adress', type=str, required=True,
                        help='ip address of the host node')

    parser.add_argument('--ngpus', default=4, type=int,
                        help='number of gpus per node')

    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to train. Look up in models.py")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--checkpoint', type=str, default=None, help="Pretrained weights")

    args = parser.parse_args()

    # Total number of gpus availabe to us.
    args.world_size = args.ngpus * args.nodes
    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_adress
    print("ip_adress is", args.ip_adress)
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    
    
    if args.checkpoint == "":
        args.checkpoint = None
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=0
    )
    torch.manual_seed(0)
        
    model = models.get_model_by_name(args.model_name)
    
    model.to(device)

    imgs_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "vkitti_2.0.3_rgb/")

    depth_root = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), config_kitti.DATA, "vkitti_2.0.3_depth/")

    annotation = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), config_kitti.COCO_ANN)


    _, data_loader_val = get_dataloaders(
        args.batch_size,
        imgs_root,
        depth_root,
        annotation,
        num_replicas=1,
        rank=0,
        split=True,
        val_size=config_kitti.VAL_SIZE,
        n_samples=config_kitti.MAX_TRAINING_SAMPLES)
    
    rmse, miou = eval_set(model, data_loader_val, args.checkpoint)

    print("rmse: ", rmse, "miou: ", miou)