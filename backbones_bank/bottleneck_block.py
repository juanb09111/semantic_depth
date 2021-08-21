# %%
import torch
from torch import nn
import torch.nn.functional as F
from .depth_wise_conv import depth_wise_conv as dwConv
import config


def sumLayer(x, y):
    return x + y

class bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv2Dexpand = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*t, kernel_size=1)
        self.batchNormExpand1 = nn.BatchNorm2d(in_channels*t)
        if config.BACKBONE_DEPTHWISE_CONV:
            print("using depthwise conv in bottleneck block")
            self.depthwiseConv = dwConv(in_channels*t, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            print("using traditional conv in bottleneck block")
            self.conv2d = nn.Conv2d(in_channels*t, in_channels*t, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNormExpand2 = nn.BatchNorm2d(in_channels*t)
        self.conv2Dsqueeze = nn.Conv2d(in_channels=in_channels*t, out_channels=out_channels, kernel_size=1)
        self.batchNormSqueeze = nn.BatchNorm2d(out_channels)
        self.sumLayer = sumLayer

    def forward(self, x):
        y = self.conv2Dexpand(x)
        y = self.batchNormExpand1(y)
        y = F.relu6(y)

        if config.BACKBONE_DEPTHWISE_CONV:
            y = self.depthwiseConv(y)
        else:
            y = self.conv2d(y)

        y = self.batchNormExpand2(y)
        y = F.relu6(y)
        y = self.conv2Dsqueeze(y)
        y = self.batchNormSqueeze(y)
        if self.in_channels == self.out_channels and self.stride == 1:
            y = self.sumLayer(x, y)
        return y

# %%