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


def RMSE(sparse_depth_gt, pred):
    mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=temp_variables.DEVICE,
                                                            dtype=torch.float64), torch.tensor((0), device=temp_variables.DEVICE, dtype=torch.float64))
    mask_gt = mask_gt.squeeze_(1)
    sparse_depth_gt = sparse_depth_gt.squeeze_(1)  # remove C dimension there's only one
    # print("shape", sparse_depth_gt.shape, mask_gt.shape)
    c = torch.tensor((1000), device=device)
    sparse_depth_gt = sparse_depth_gt*c
    pred = pred*c
    criterion = nn.MSELoss()
    res = torch.sqrt(criterion(sparse_depth_gt*mask_gt, pred*mask_gt))
    return res


def mIOU(label, pred):
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




def __results_to_json(model, data_loader_val):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    res = []
    rmse_arr = []
    iou_arr = []
    for images, anns, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, _, _ in data_loader_val:
        # for images, anns, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, _ in data_loader_val:
        imgs = list(img for img in images)
        lidar_fov = list(lid_fov for lid_fov in lidar_fov)
        masks = list(mask for mask in masks)
        sparse_depth = list(sd for sd in sparse_depth)
        k_nn_indices = list(k_nn for k_nn in k_nn_indices)
        sparse_depth_gt = list(sp_d for sp_d in sparse_depth_gt)
        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in anns]

        imgs = tensorize_batch(imgs, device)
        lidar_fov = tensorize_batch(lidar_fov, device, dtype=torch.float)
        masks = tensorize_batch(masks, device, dtype=torch.bool)
        sparse_depth = tensorize_batch(sparse_depth, device)
        k_nn_indices = tensorize_batch(k_nn_indices, device, dtype=torch.long)
        sparse_depth_gt = tensorize_batch(
            sparse_depth_gt, device, dtype=torch.float)

        model.eval()

        with torch.no_grad():
            outputs = model(imgs,  sparse_depth,
                            masks,
                            lidar_fov,
                            k_nn_indices,
                            anns=annotations)
            # _, outputs = model(imgs, sparse_depth, masks, lidar_fov, k_nn_indices, anns=annotations, sparse_depth_gt=sparse_depth_gt)

            for idx, out in enumerate(outputs):
                
                # Calculate rmse
                # print("shape", sparse_depth_gt.shape, out["depth"].shape, anns[idx]["semantic_mask"].shape, out["semantic_logits"].shape)
                rmse = RMSE(sparse_depth_gt, out["depth"])

                rmse_arr.append(rmse.cpu().data.numpy())

                #Calculate miou
                semantic_mask = anns[idx]["semantic_mask"]
                semantic_logits = out["semantic_logits"]
                iou = mIOU(semantic_mask, semantic_logits)
                iou_arr.append(iou)

        torch.cuda.empty_cache()
    return np.mean(rmse_arr), np.mean(iou_arr)


def __export_res(model, data_loader_val):
    rmse, iou = __results_to_json(model, data_loader_val)

    
    return rmse, iou





def get_mIoU_effps(model, data_loader_val):

    iou_list = []
    for images, anns, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, _, img_name in data_loader_val:

        imgs = list(img for img in images)

        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in anns]

        imgs = tensorize_batch(imgs, device)

        model.eval()

        with torch.no_grad():
            # _, outputs = model(imgs, sparse_depth, masks, lidar_fov, k_nn_indices, anns=annotations, sparse_depth_gt=sparse_depth_gt)
            outputs = model(imgs, anns=annotations)
            # outputs = model(images)

            for idx, output in enumerate(outputs):

                label = anns[idx]["semantic_mask"]
                pred = output["semantic_logits"]
                iou = mIOU(label, pred)
                iou_list.append(iou)

                label = label.cpu().numpy()
                pred = F.softmax(pred, dim=0)
                pred = torch.argmax(pred, dim=0).squeeze(1)
                pred = pred.cpu().numpy()
                # ---------------------------------------
                f, (ax1, ax2, ax3) = plt.subplots(3, 1)

                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                ax1.imshow(imgs[idx].permute(1, 2, 0).cpu().numpy())
                # print('img')
                # plt.show()
                ax1.axis('off')

                ax2.imshow(label)
                # print('out')
                # plt.show()
                ax2.axis('off')

                ax3.imshow(pred)
                # print('out')
                # plt.show()
                ax3.axis('off')

                # print(out.max(), out.min())

                f.savefig('{}semseg_effps_no_instance/{}_iou={}.png'.format(
                    constants.EFUSION_RESULTS, img_name[idx], iou))
                plt.close(f)

    return np.mean(iou_list)







def evaluate(all_cat, things_cat, model=None, weights_file=None, data_loader_val=None, train_res_file=None):
    """This function performs AP evaluation using coco_eval"""

    if train_res_file is None:
        train_res_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.RES_LOC, constants.EVAL_RES_FILENAME)

    # train_res_file = train_res_file if config.DATA_LOADER is None else config.DATA_LOADER

    if weights_file is None and model is None:
        # Get model weights from config
        weights_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), config_kitti.MODEL_WEIGHTS_FILENAME)
    print("weights_file", weights_file)
    if model is None:
        # Get model corresponding to the one selected in config
        model = models.get_model_by_name(config_kitti.MODEL)
        # Load model weights
        print("loading wights...")
        model.load_state_dict(torch.load(weights_file))
    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Empty cache
    torch.cuda.empty_cache()
    # Model to device
    model.to(device)

    if data_loader_val is None:
        # Data loader is in constants.DATA_LOADERS_LOC/constants.DATA_LOADER_VAL_FILENAME by default
        data_loader_val = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_VAL_FILENAME_OBJ)

        # If DATA_LOADER is None in config then use default dataloader=data_loader_val as defined above
        data_loader_val = data_loader_val if config_kitti.DATA_LOADER is None else config_kitti.DATA_LOADER
        print("data_loader_val", data_loader_val)

        # Load dataloader
        data_loader_val = torch.load(data_loader_val)
        print(len(data_loader_val))


    # Export the predictions as a json file
    rmse, average_iou = __export_res(model, data_loader_val)



    return average_iou, rmse

# if __name__ == "__main__":
#     wieghts_file = "tmp/models/EfficientPS_weights_loss_0.5110576054896181.pth"
#     evaluate(model=None, weights_file=wieghts_file, data_loader_val=None)
