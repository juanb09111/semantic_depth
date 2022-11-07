#%%
import torch
from torch import nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn 
import config_kitti

class MaskRCNN(nn.Module):
    def __init__(self, num_ins_classes):
        super().__init__()
        # min_size, max_size = config_kitti.MIN_SIZE, config_kitti.MAX_SIZE
        # self.mask_rcnn = maskrcnn_resnet50_fpn(num_classes=num_ins_classes + 1, min_size=min_size, max_size=max_size)
        self.mask_rcnn = maskrcnn_resnet50_fpn(num_classes=num_ins_classes + 1, pretrained_backbone=False)
        for module in self.children():
            if self.training:
                module.training = True
            else:
                module.training = False

    def forward(self, images, anns=None):


        if self.training:
            maskrcnn_losses = self.mask_rcnn(images, anns)

        else:
            maskrcnn_results = self.mask_rcnn(images)
            
  
        if self.training:

            return {**maskrcnn_losses}

        else:
            # print(maskrcnn_results)
            return [{**maskrcnn_results[idx]} for idx, _ in enumerate(images)]


# model_s = MaskRCNN_backbone()

# images = torch.rand((2, 3, 1024, 2048))

# P4, P8, P16, P32 = model_s(images)
# print(P4.shape)
# %%
