#%%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from .LSFE import LSFE 
from .DPC import DPC 
from .MC import MC
#%%
class refine_head(nn.Module):
    def __init__(self, num_classes, output_resol):
        super().__init__()
        
        self.output_resol = output_resol


        # self.out_conv_depth = nn.Conv2d(num_classes + 1, num_classes, kernel_size=1)

        self.out_conv_depth_1 = nn.Sequential(
            nn.Conv2d(num_classes+1, 256, kernel_size=3),
            nn.BatchNorm2d(256)
        )

        self.out_conv_depth_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256)
        )

        self.out_conv_depth_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256)
        )

        self.out_conv_depth_4 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes)
        )

    def forward(self, x, depth_map):
       

        #----------------------------
        x = torch.cat((x, depth_map), dim=1)

        x = F.leaky_relu(self.out_conv_depth_1(x))
        x = F.leaky_relu(self.out_conv_depth_2(x))

        x = F.leaky_relu(self.out_conv_depth_3(x))
        x = F.leaky_relu(self.out_conv_depth_4(x))

        x = F.interpolate(x, size=self.output_resol, mode="bilinear")
        return x


