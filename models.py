import os.path
import json
import constants
from models_bank.efusion_ps_v3_depth_head_2 import EfusionPS as EfusionPS_V3_depth
from models_bank.fusenet import FuseNet
import config_kitti


def get_model_by_name(model_name):



    
    if model_name is "EfusionPS_V3_depth":
        return EfusionPS_V3_depth(config_kitti.BACKBONE,
                         config_kitti.BACKBONE_OUT_CHANNELS,
                         config_kitti.K_NUMBER,
                         config_kitti.NUM_THING_CLASSES,
                         config_kitti.NUM_STUFF_CLASSES,
                         config_kitti.CROP_OUTPUT_SIZE,
                         min_size=config_kitti.MIN_SIZE,
                         max_size=config_kitti.MAX_SIZE,
                         n_number=config_kitti.N_NUMBER)

    if model_name is "FuseNet":
        return FuseNet(config_kitti.K_NUMBER,
                         n_number=config_kitti.N_NUMBER)



