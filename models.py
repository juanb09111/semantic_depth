import os.path
import json
import constants
from models_bank.efusion_ps_v3_depth_head_2 import EfusionPS as EfusionPS_V3_depth
from models_bank.fusenet import FuseNet
from models_bank.sem_seg_net import SemsegNet
from models_bank.sem_seg_depth import Semseg_Depth
from models_bank.sem_seg_depth_v2 import Semseg_Depth as Semseg_Depth_v2
from models_bank.sem_seg_depth_v3 import Semseg_Depth as Semseg_Depth_v3
from models_bank.sem_seg_net_depth_input import SemsegNet_DepthInput
import config_kitti

MODELS = ["FuseNet",
          "SemsegNet",
          "Semseg_Depth",
          "Semseg_Depth_v2",
          "Semseg_Depth_v3",
          "SemsegNet_DepthInput"]


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

    if model_name is "SemsegNet":
        return SemsegNet(config_kitti.BACKBONE_OUT_CHANNELS,
                         config_kitti.NUM_THING_CLASSES,
                         config_kitti.NUM_STUFF_CLASSES,
                         config_kitti.CROP_OUTPUT_SIZE)

    if model_name is "SemsegNet_DepthInput":
        return SemsegNet(config_kitti.BACKBONE_OUT_CHANNELS,
                         config_kitti.NUM_THING_CLASSES,
                         config_kitti.NUM_STUFF_CLASSES,
                         config_kitti.CROP_OUTPUT_SIZE)


    if model_name is "Semseg_Depth":
        return Semseg_Depth(config_kitti.K_NUMBER,
                            config_kitti.BACKBONE_OUT_CHANNELS,
                            config_kitti.NUM_THING_CLASSES,
                            config_kitti.NUM_STUFF_CLASSES,
                            config_kitti.CROP_OUTPUT_SIZE,
                            n_number=config_kitti.N_NUMBER)

    if model_name is "Semseg_Depth_v2":
        return Semseg_Depth_v2(config_kitti.K_NUMBER,
                               config_kitti.BACKBONE_OUT_CHANNELS,
                               config_kitti.NUM_THING_CLASSES,
                               config_kitti.NUM_STUFF_CLASSES,
                               config_kitti.CROP_OUTPUT_SIZE,
                               n_number=config_kitti.N_NUMBER)

    if model_name is "Semseg_Depth_v3":
        return Semseg_Depth_v3(config_kitti.K_NUMBER,
                               config_kitti.BACKBONE_OUT_CHANNELS,
                               config_kitti.NUM_THING_CLASSES,
                               config_kitti.NUM_STUFF_CLASSES,
                               config_kitti.CROP_OUTPUT_SIZE,
                               n_number=config_kitti.N_NUMBER)
