import config_kitti
import temp_variables
import constants
import models

from utils.tensorize_batch import tensorize_batch

import os
import os.path
from pathlib import Path
import torch
from torch import nn

from datetime import datetime

import matplotlib.pyplot as plt


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

temp_variables.DEVICE = device
torch.cuda.empty_cache()

def RMSE(sparse_depth_gt,pred):
    mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=temp_variables.DEVICE, dtype=torch.float64), torch.tensor((0), device=temp_variables.DEVICE, dtype=torch.float64))
    mask_gt = mask_gt.squeeze_(1)
    sparse_depth_gt = sparse_depth_gt.squeeze_(1) # remove C dimension there's only one
    c = torch.tensor((1000), device=device)
    sparse_depth_gt = sparse_depth_gt*c
    pred = pred*c
    criterion = nn.MSELoss()
    res = torch.sqrt(criterion(sparse_depth_gt, pred*mask_gt))
    return res


def predict(model,
            data_loader_val,
            weights_file,
            folder):

    # Create folde if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    # load weights
    print(weights_file)
    model.load_state_dict(torch.load(weights_file))
    # move model to the right device
    model.to(device)

    model.eval()

    with torch.no_grad():
        
        for imgs, anns, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, _  in data_loader_val:

            imgs = list(img for img in imgs)
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
            k_nn_indices = tensorize_batch(
                k_nn_indices, device, dtype=torch.long)
            sparse_depth_gt = tensorize_batch(
                sparse_depth_gt, device, dtype=torch.float)

            outputs = model(imgs,  sparse_depth,
                            masks,
                            lidar_fov,
                            k_nn_indices,
                            anns=annotations)
            
            for idx in range(len(outputs)):

                out_depth = outputs[idx]["depth"]

                ## --------------------------------------
                # rmse = RMSE(sparse_depth_gt, out_depth)

                mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=temp_variables.DEVICE, dtype=torch.float64), torch.tensor((0), device=temp_variables.DEVICE, dtype=torch.float64))
                mask_gt = mask_gt.squeeze_(1)
                sparse_depth_gt = sparse_depth_gt.squeeze_(1) # remove C dimension there's only one
                c = torch.tensor((1000), device=device)
                sparse_depth_gt = sparse_depth_gt*c
                pred = out_depth*c
                criterion = nn.MSELoss()
                rmse = torch.sqrt(criterion(sparse_depth_gt, pred*mask_gt))
                ##-----------------------------------------
                out = outputs[idx]["depth"].cpu().numpy()
                img = imgs[idx]

                f, (ax1, ax2) = plt.subplots(2, 1)

                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                ax1.imshow(img.permute(1, 2, 0).cpu().numpy())
                # print('img')
                # plt.show()
                ax1.axis('off')

                ax2.imshow(out)
                # print('out')
                # plt.show()
                ax2.axis('off')

                # print(out.max(), out.min())

                f.savefig('{}/rmse_{}.png'.format(folder, rmse))
                plt.close(f)


            # for idx in range(outputs.shape[0]):

            #     rmse = RMSE(sparse_depth_gt, outputs[idx])
                
            #     out = outputs[idx].cpu().numpy()
            #     img = imgs[idx]

            #     f, (ax1, ax2) = plt.subplots(2, 1)

            #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
            #                         hspace=0, wspace=0)
            #     plt.margins(0, 0)
            #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
            #     plt.gca().yaxis.set_major_locator(plt.NullLocator())

            #     ax1.imshow(img.permute(1, 2, 0).cpu().numpy())
            #     # print('img')
            #     # plt.show()
            #     ax1.axis('off')

            #     ax2.imshow(out)
            #     # print('out')
            #     # plt.show()
            #     ax2.axis('off')

            #     # print(out.max(), out.min())

            #     f.savefig('{}/rmse_{}.png'.format(folder, rmse))
            #     plt.close(f)
                


if __name__ == "__main__":
    torch.cuda.empty_cache()

    data_loader_val = torch.load(config_kitti.DATA_LOADER_VAL_FILENAME)

    model = models.get_model_by_name(config_kitti.MODEL)

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    folder = '{}_{}_eval_results_depth_completion_{}'.format(constants.INFERENCE_RESULTS,
                                                             config_kitti.MODEL, timestamp)

    predict(model, data_loader_val, config_kitti.MODEL_WEIGHTS_FILENAME, folder)
