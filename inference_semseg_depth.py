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
                mask_output.unsqueeze(0),device).squeeze(0)/255
                mask_output= mask_output.cpu().permute(1, 2, 0).numpy()

                rgb = imgs[idx].cpu().permute(1, 2, 0).numpy()

                loc = "{}/{}".format(constants.INFERENCE_RESULTS, "Semseg_Depth_v4")
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
    
    # Get model according to config
    print(models.get_model_by_name, args.model_name)
    model = models.get_model_by_name("Semseg_Depth_v4")
    
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

    inference_sem_seg_depth(model, data_loader_val, args.checkpoint)

