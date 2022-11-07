# %%
import os
import os.path
import sys
import torch.multiprocessing as mp
from argparse import ArgumentParser
from train import train_fusenet
from train import train_fusenet_v2
from train import train_sem_seg_net
from train import train_semseg_depth_v2
from train import train_semseg_depth_v3
from train import train_semseg_depth_v4
from train import train_semseg_depth
from train import train_semseg_depth_input
from train import train_semseg_depth_v2_loss_sum
from train import train_panoptic
from train import train_mask_rcnn
from train import train_panoptic_depth
from train import train_mask_rcnn_ytvos
from models import MODELS
# # from ignite.contrib.handlers.param_scheduler import PiecewiseLinear

def get_train_loop(model_name, dataset="vkitti"):
    # if model_name == "FuseNet":
    #     return train_fusenet.train
    # if model_name == "FuseNet_v2":
    #     return train_fusenet_v2.train
    # if model_name == "SemsegNet":
    #     return train_sem_seg_net.train
    # if model_name == "Semseg_Depth":
    #     return train_semseg_depth.train
    # if model_name == "Semseg_Depth_v2":
    #     return train_semseg_depth_v2.train
    # if model_name == "Semseg_Depth_v3":
    #     return train_semseg_depth_v3.train
    # if model_name == "Semseg_Depth_v4":
    #     return train_semseg_depth_v4.train
    # if model_name == "SemsegNet_DepthInput":
    #     return train_semseg_depth_input.train
    # if model_name == "Semseg_Depth_v2_loss_sum":
    #     return train_semseg_depth_v2_loss_sum.train
    # if model_name == "PanopticSeg":
    #     return train_panoptic.train
    if model_name == "MaskRcnn":
        if dataset == "vkitti":
            return train_mask_rcnn.train
        elif dataset == "ytvos":
            return train_mask_rcnn_ytvos.train
    if model_name == "PanopticDepth":
        return train_panoptic_depth.train
          

if __name__ == "__main__":

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

    parser.add_argument('--ann_file', type=str, required=True, help="Annotation json file")
    parser.add_argument('--data', type=str, required=True, help="rgb folder")
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    

    parser.add_argument('--dataset', type=str, default="vkitti", required=False, help="dataset name")
    parser.add_argument('--n_samples', default=None, required=False, help="Number of training samples")
    parser.add_argument('--val_size', default=0.2, type=int, required=False, help="validation set size")
    parser.add_argument('--checkpoint', type=str, default=None, help="Pretrained weights")
    parser.add_argument('--lr', type=float, default=0.0025, help="learning rate")

    args = parser.parse_args()
    if args.checkpoint == "":
        args.checkpoint = None
    
    if args.n_samples == "":
        args.n_samples = None

    print(args)
    if args.model_name not in MODELS:
        raise ValueError("model_name must be one of: ", MODELS)
    train_loop = get_train_loop(args.model_name, args.dataset)

    # Total number of gpus availabe to us.
    args.world_size = args.ngpus * args.nodes
    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_adress
    print("ip_adress is", args.ip_adress)
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(train_loop, nprocs=args.ngpus, args=(args,))

