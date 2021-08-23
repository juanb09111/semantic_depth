import constants
from utils.tensorize_batch import tensorize_batch
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from backbones_bank.tunned_maskrcnn.utils.backbone_utils import resnet_fpn_backbone

import config_kitti
import temp_variables




class SemsegNet(nn.Module):
    def __init__(self, backbone_out_channels,
                 num_ins_classes,
                 num_sem_classes,
                 original_image_size):
        super(SemsegNet, self).__init__()



        self.backbone = resnet_fpn_backbone('resnet50', True)
        # backbone.body.conv1 = nn.Conv2d(48, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


        self.semantic_head = sem_seg_head(
            backbone_out_channels,
            num_ins_classes + num_sem_classes + 1, original_image_size, depthwise_conv=config_kitti.SEMANTIC_HEAD_DEPTHWISE_CONV)



        for module in self.children():
            if self.training:
                module.training = True
            else:
                module.training = False

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self, images, anns=None):

        
        losses = {}
        semantic_logits = []

        backbone_feat = self.backbone(images)

        P4, P8, P16, P32 = backbone_feat['0'], backbone_feat['1'], backbone_feat['2'], backbone_feat['3']
        
        semantic_logits = self.semantic_head(P4, P8, P16, P32)
  
        if self.training:

            semantic_masks = list(
                map(lambda ann: ann['semantic_mask'], anns))
            semantic_masks = tensorize_batch(
                semantic_masks, temp_variables.DEVICE)

            losses["semantic_loss"] = F.cross_entropy(
                semantic_logits, semantic_masks.long())

            return losses

        else:
            return [{ 'semantic_logits': semantic_logits[idx]} for idx, _ in enumerate(images)]


