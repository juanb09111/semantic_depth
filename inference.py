# %%
import os
import os.path
import sys
import torch.multiprocessing as mp
from argparse import ArgumentParser
from inference_scripts import inference_depth_completion
from inference_scripts import inference_Fusenet
from inference_scripts import inference_panoptic
from inference_scripts import inference_instance
from inference_scripts import inference_semseg_depth
from inference_scripts import inference_semseg_net
from models import MODELS
# # from ignite.contrib.handlers.param_scheduler import PiecewiseLinear

def get_inference_loop(model_name):
    
    if model_name == "Semseg_Depth_v4":
        return inference_semseg_depth.inference
    if model_name == "PanopticSeg":
        return inference_panoptic.inference
    if model_name == "MaskRcnn":
        return inference_instance.inference
          

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
    parser.add_argument('--checkpoint', type=str, required=True, help="Pretrained weights")
    parser.add_argument('--categories_json', type=str, required=True, help="Categories, COCO format json file. No annotations required, categories only.")
    parser.add_argument('--dst', type=str, required=True, help="Output folder")
    parser.add_argument('--data_source', type=str, default=None, help="Pytorch Dataloader. If None dataloader is required")
 
   

    args = parser.parse_args()

    if ".pth" in args.data_source:
        args.data_folder = None
        args.dataloader = args.data_source
    else:
        args.dataloader = None
        args.data_folder = args.data_source
    print(args)
    if args.checkpoint == "":
        args.checkpoint = None

    if args.model_name not in MODELS:
        raise ValueError("model_name must be one of: ", MODELS)
    inference_loop = get_inference_loop(args.model_name)

    # Total number of gpus availabe to us.
    args.world_size = args.ngpus * args.nodes
    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_adress
    print("ip_adress is", args.ip_adress)
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(inference_loop, nprocs=args.ngpus, args=(args,))

