import config_kitti

import constants
import models
import numpy as np
from utils.tensorize_batch import tensorize_batch

import os
import os.path
from pathlib import Path
import torch
from torch import nn

from datetime import datetime

import matplotlib.pyplot as plt



torch.cuda.empty_cache()


def RMSE(sparse_depth_gt, pred, device):
    with torch.no_grad():
        mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=device,
                                                                dtype=torch.float64), torch.tensor((0), device=device, dtype=torch.float64))
        mask_gt = mask_gt.squeeze_(1)

        sparse_depth_gt = sparse_depth_gt.squeeze_(0)  # remove C dimension there's only one

        

        c = torch.tensor((1000), device=device)
        sparse_depth_gt = sparse_depth_gt*c
        pred = pred*c

        criterion = nn.MSELoss()
        # print("VAL", pred.shape, sparse_depth_gt.shape)
        res = torch.sqrt(criterion(sparse_depth_gt*mask_gt, pred*mask_gt))

    
    return res


def eval_depth(model, data_loader_val, weights_file, device):


    # load weights
    print("eval depth completion weights: ", weights_file)
    model.load_state_dict(torch.load(weights_file)["state_dict"])
    # move model to the right device
    model.to(device)

    rmse_arr = []

    model.eval()

    with torch.no_grad():

        for imgs, ann, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, sparse_depth_gt_full, _ in data_loader_val:

            imgs = list(img for img in imgs)
            lidar_fov = list(lid_fov for lid_fov in lidar_fov)
            masks = list(mask for mask in masks)
            sparse_depth = list(sd for sd in sparse_depth)
            k_nn_indices = list(k_nn for k_nn in k_nn_indices)
            sparse_depth_gt = list(sp_d for sp_d in sparse_depth_gt)
            sparse_depth_gt_full = list(sp_d_full for sp_d_full in sparse_depth_gt_full)

            imgs = tensorize_batch(imgs, device)
            lidar_fov = tensorize_batch(lidar_fov, device, dtype=torch.float)
            masks = tensorize_batch(masks, device, dtype=torch.bool)
            sparse_depth = tensorize_batch(sparse_depth, device)
            k_nn_indices = tensorize_batch(
                k_nn_indices, device, dtype=torch.long)
            sparse_depth_gt = tensorize_batch(
                sparse_depth_gt, device, dtype=torch.float) #BxCxHW
            sparse_depth_gt_full = tensorize_batch(
                sparse_depth_gt_full, device, dtype=torch.float)
            
            annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in ann]
        
            semantic_masks = list(map(lambda ann: ann['semantic_mask'], annotations))
            
            semantic_masks = tensorize_batch(semantic_masks, device)


            outputs = model(imgs,  sparse_depth,
                            masks,
                            lidar_fov,
                            k_nn_indices,
                            semantic_masks,
                            sparse_depth_gt=None)

            for idx in range(len(outputs)):

                # print(sparse_depth_gt.shape)
                out_depth = outputs[idx]["depth"]
                
                # --------------------------------------
                rmse = RMSE(sparse_depth_gt[idx], out_depth, device)
                rmse_arr.append(rmse.cpu().data.numpy())
                # print(sparse_depth_gt.shape)
                # -----------------------------------------
    
    return np.mean(rmse_arr), imgs[0], sparse_depth_gt[0], sparse_depth_gt_full[0].squeeze_(0), outputs[0]["depth"]
