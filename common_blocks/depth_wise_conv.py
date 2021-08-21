# %%
import torch
from torch import nn
import torch.nn.functional as F


class depth_wise_conv(nn.Module):
    def __init__(self, nin, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, stride=stride, dilation=dilation)

    def forward(self, x):
        x = self.depthwise(x)
        return x
