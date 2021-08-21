#%%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math

class MaskRCNN_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # self.resolution = (math.ceil(1024*original_aspect_ratio), 1024)
        maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.backbone = maskrcnn.backbone
    def forward(self, x):
        
        # x = F.interpolate(x, size=self.resolution)
        x = self.backbone(x)

        return x["0"], x["1"], x["2"], x["3"]

# model_s = MaskRCNN_backbone()

# images = torch.rand((2, 3, 1024, 2048))

# P4, P8, P16, P32 = model_s(images)
# print(P4.shape)
# %%
