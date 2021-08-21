import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.depth_wise_conv import depth_wise_conv
#%%
class LSFE(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise_conv=True):
        super().__init__()

        self.depthwise_conv = depthwise_conv

        if depthwise_conv:
            print("using depthwise conv in LSFE module")
            self.conv1_depth_wise_conv = nn.Sequential(
                depth_wise_sep_conv(in_channels, out_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(out_channels)
            )

            self.conv2_depth_wise_conv = nn.Sequential(
                depth_wise_conv(out_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            print("using traditional conv in LSFE module")
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
            x = F.leaky_relu(self.conv1_depth_wise_conv(x))
            x = F.leaky_relu(self.conv2_depth_wise_conv(x))
        else:
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))


        return x


# images = torch.rand((2, 256, 64, 64))
# model = LSFE(256, 128)
# print(model)
# x = model(images)
# print(x.shape)