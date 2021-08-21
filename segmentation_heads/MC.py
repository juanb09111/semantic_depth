import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.depth_wise_conv import depth_wise_conv
#%%
class MC(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise_conv=True):
        super().__init__()

        self.depthwise_conv = depthwise_conv

        if depthwise_conv:
            print("using depthwise conv in MC module")
            # depthwise conv
            self.depthwise_conv1 = nn.Sequential(
                depth_wise_sep_conv(in_channels, out_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(out_channels)
            )

            self.depthwise_conv2 = nn.Sequential(
                depth_wise_conv(out_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            print("using traditional conv in MC module")
            # traditional conv
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(out_channels)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        if self.depthwise_conv:
            x = F.leaky_relu(self.depthwise_conv1(x))
            x = F.leaky_relu(self.depthwise_conv2(x))
            
        else:
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
        
        # MC module upsamples by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        return x


# images = torch.rand((2, 256, 64, 64))
# model = MC(256, 128)
# print(model)
# x = model(images)
# print(x.shape)