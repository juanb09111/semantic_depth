#%%
import torch
from torch import nn
import torchvision

class MaskRCNN_backbone(nn.Module):
    def __init__(self, backbone_out_channels):
        super().__init__()
        # self.resolution = (math.ceil(1024*original_aspect_ratio), 1024)
        maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        self.backbone = maskrcnn.backbone
        self.out_channels = backbone_out_channels
    def forward(self, x):
        
        # x = F.interpolate(x, size=self.resolution)
        x = self.backbone(x)

        return x["0"], x["1"], x["2"], x["3"]

# model_s = MaskRCNN_backbone()

# images = torch.rand((2, 3, 1024, 2048))

# P4, P8, P16, P32 = model_s(images)
# print(P4.shape)
# %%
