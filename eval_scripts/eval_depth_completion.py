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

        #visualize

        # out_copy = pred
        # plt.imshow(out_copy.cpu().detach().numpy())
        # # print('out_copy', out.max(), out.min())
        # plt.show()

        # sparse_depth_gt_copy = sparse_depth_gt
        # # # print("shape", sparse_depth_gt_copy.cpu().detach().numpy().shape)
        # plt.imshow(sparse_depth_gt_copy.cpu().detach().numpy())
        # print('sparse_depth_gt_copy' , sparse_depth_gt.max(), sparse_depth_gt.min())
        # print('pred' , pred.max(), pred.min())
        # plt.show()

        

        # mask_gt_copy = mask_gt[0]
        # # # print("shape", sparse_depth_gt_copy.cpu().detach().numpy().shape)
        # plt.imshow(mask_gt_copy.cpu().detach().numpy())
        # # print('mask_gt_copy')
        # plt.show()
        #---

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

        for imgs, _, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, sparse_depth_gt_full, _ in data_loader_val:

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
            


            outputs = model(imgs,  sparse_depth,
                            masks,
                            lidar_fov,
                            k_nn_indices,
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


# if __name__ == "__main__":
#     torch.cuda.empty_cache()

#     data_loader_val = torch.load(config_kitti.DATA_LOADER_VAL_FILENAME)

#     model = models.get_model_by_name(config_kitti.MODEL)
   
#     eval_depth(model, data_loader_val, config_kitti.MODEL_WEIGHTS_FILENAME)
