import os.path
import json
import constants
from models_bank.efusion_ps_v3_depth_head_2 import EfusionPS as EfusionPS_V3_depth
from models_bank.fusenet import FuseNet
from models_bank.fusenet_v2 import FuseNet_v2
from models_bank.sem_seg_net import SemsegNet
from models_bank.sem_seg_depth import Semseg_Depth
from models_bank.sem_seg_depth_v2 import Semseg_Depth as Semseg_Depth_v2
from models_bank.sem_seg_depth_v2_loss_sum import Semseg_Depth as Semseg_Depth_v2_loss_sum
from models_bank.sem_seg_depth_v4 import Semseg_Depth as Semseg_Depth_v4
from models_bank.sem_seg_depth_v3 import Semseg_Depth as Semseg_Depth_v3
from models_bank.sem_seg_net_depth_input import SemsegNet_DepthInput
from models_bank.panoptic_seg import PanopticSeg
from models_bank.panoptic_depth import PanopticDepth

import config_kitti

MODELS = ["FuseNet",
          "SemsegNet",
          "Semseg_Depth",
          "Semseg_Depth_v2",
          "Semseg_Depth_v2_loss_sum",
          "Semseg_Depth_v3",
          "Semseg_Depth_v4",
          "SemsegNet_DepthInput",
          "FuseNet_v2",
          "PanopticSeg",
          "PanopticDepth"]


def get_model_by_name(model_name):
    if model_name == "EfusionPS_V3_depth":
        return EfusionPS_V3_depth(config_kitti.BACKBONE,
                                  config_kitti.BACKBONE_OUT_CHANNELS,
                                  config_kitti.K_NUMBER,
                                  config_kitti.NUM_THING_CLASSES,
                                  config_kitti.NUM_STUFF_CLASSES,
                                  config_kitti.CROP_OUTPUT_SIZE,
                                  min_size=config_kitti.MIN_SIZE,
                                  max_size=config_kitti.MAX_SIZE,
                                  n_number=config_kitti.N_NUMBER)

    if model_name == "FuseNet":
        return FuseNet(config_kitti.K_NUMBER,
                       n_number=config_kitti.N_NUMBER)

    if model_name == "FuseNet_v2":
        return FuseNet_v2(config_kitti.K_NUMBER,
                          n_number=config_kitti.N_NUMBER)

    if model_name == "SemsegNet":
        return SemsegNet(config_kitti.BACKBONE_OUT_CHANNELS,
                         config_kitti.NUM_THING_CLASSES,
                         config_kitti.NUM_STUFF_CLASSES,
                         config_kitti.CROP_OUTPUT_SIZE)

    if model_name == "SemsegNet_DepthInput":
        return SemsegNet_DepthInput(config_kitti.BACKBONE_OUT_CHANNELS,
                                    config_kitti.NUM_THING_CLASSES,
                                    config_kitti.NUM_STUFF_CLASSES,
                                    config_kitti.CROP_OUTPUT_SIZE)

    if model_name == "Semseg_Depth":
        return Semseg_Depth(config_kitti.K_NUMBER,
                            config_kitti.BACKBONE_OUT_CHANNELS,
                            config_kitti.NUM_THING_CLASSES,
                            config_kitti.NUM_STUFF_CLASSES,
                            config_kitti.CROP_OUTPUT_SIZE,
                            n_number=config_kitti.N_NUMBER)

    if model_name == "Semseg_Depth_v2":
        return Semseg_Depth_v2(config_kitti.K_NUMBER,
                               config_kitti.BACKBONE_OUT_CHANNELS,
                               config_kitti.NUM_THING_CLASSES,
                               config_kitti.NUM_STUFF_CLASSES,
                               config_kitti.CROP_OUTPUT_SIZE,
                               n_number=config_kitti.N_NUMBER)

    if model_name == "Semseg_Depth_v4":
        return Semseg_Depth_v4(config_kitti.K_NUMBER,
                               config_kitti.BACKBONE_OUT_CHANNELS,
                               config_kitti.NUM_THING_CLASSES,
                               config_kitti.NUM_STUFF_CLASSES,
                               config_kitti.CROP_OUTPUT_SIZE,
                               n_number=config_kitti.N_NUMBER)

    if model_name == "Semseg_Depth_v2_loss_sum":
        return Semseg_Depth_v2_loss_sum(config_kitti.K_NUMBER,
                                        config_kitti.BACKBONE_OUT_CHANNELS,
                                        config_kitti.NUM_THING_CLASSES,
                                        config_kitti.NUM_STUFF_CLASSES,
                                        config_kitti.CROP_OUTPUT_SIZE,
                                        n_number=config_kitti.N_NUMBER)

    if model_name == "Semseg_Depth_v3":
        return Semseg_Depth_v3(config_kitti.K_NUMBER,
                               config_kitti.BACKBONE_OUT_CHANNELS,
                               config_kitti.NUM_THING_CLASSES,
                               config_kitti.NUM_STUFF_CLASSES,
                               config_kitti.CROP_OUTPUT_SIZE,
                               n_number=config_kitti.N_NUMBER)

    if model_name == "PanopticSeg":
        return PanopticSeg(config_kitti.BACKBONE_OUT_CHANNELS,
                           config_kitti.NUM_THING_CLASSES,
                           config_kitti.NUM_STUFF_CLASSES,
                           config_kitti.CROP_OUTPUT_SIZE,
                           pre_trained_backboned=config_kitti.PRE_TRAINED_BACKBONE,
                           backbone_name=config_kitti.BACKBONE)

    if model_name == "PanopticDepth":
        return PanopticDepth(config_kitti.K_NUMBER,
                             config_kitti.BACKBONE_OUT_CHANNELS,
                             config_kitti.NUM_THING_CLASSES,
                             config_kitti.NUM_STUFF_CLASSES,
                             config_kitti.CROP_OUTPUT_SIZE,
                             n_number=config_kitti.N_NUMBER,
                             pre_trained_backboned=config_kitti.PRE_TRAINED_BACKBONE,
                             backbone_name=config_kitti.BACKBONE)
