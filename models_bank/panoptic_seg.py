import constants
from utils.tensorize_batch import tensorize_batch
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


from models_bank.maskrcnn.detection import maskrcnn_resnet50_fpn
from models_bank.maskrcnn.detection.backbone_utils import resnet_fpn_backbone


import config_kitti


#%%


class PanopticSeg(nn.Module):
    def __init__(self, backbone_out_channels,
                 num_ins_classes,
                 num_sem_classes,
                 input_image_size,
                 pre_trained_backboned=False,
                 backbone_name="resnet50"):
        super(PanopticSeg, self).__init__()


        min_size , max_size = input_image_size
        
        self.backbone = resnet_fpn_backbone(backbone_name, pre_trained_backboned)
        # backbone.body.conv1 = nn.Conv2d(48, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.mask_rcnn = maskrcnn_resnet50_fpn(
            pretrained=False, backbone=self.backbone, num_classes=num_ins_classes + 1, min_size=min_size, max_size=max_size)
        
        self.semantic_head = sem_seg_head(
            backbone_out_channels,
            num_ins_classes + num_sem_classes + 1, input_image_size, depthwise_conv=config_kitti.SEMANTIC_HEAD_DEPTHWISE_CONV)



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

        if self.training:
            maskrcnn_losses, backbone_feat = self.mask_rcnn(images, anns)

        else:
            maskrcnn_results, backbone_feat = self.mask_rcnn(images)
            

        P4, P8, P16, P32 = backbone_feat['0'], backbone_feat['1'], backbone_feat['2'], backbone_feat['3']
        
        semantic_logits = self.semantic_head(P4, P8, P16, P32)
  
        if self.training:

            device = semantic_logits.get_device()
            
            semantic_masks = list(map(lambda ann: ann['semantic_mask'], anns))
            semantic_masks = tensorize_batch(semantic_masks, device)

            losses["semantic_loss"] = F.cross_entropy(
                semantic_logits, semantic_masks.long())

            return {**losses, **maskrcnn_losses}

        else:
            return [{**maskrcnn_results[idx], 'semantic_logits': semantic_logits[idx]} for idx, _ in enumerate(images)]

