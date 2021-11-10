# %%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from .bottleneck_block import bottleneck_block as MBConv
import math

"""The first layer of each
sequence has a stride s and all others use stride 1"""

DEPTHS = [1, 1, 2, 2, 3, 3, 4, 1, 1]


class efficient_net(nn.Module):
    def __init__(self, in_channels, width, depth, r):
        super().__init__()

        self.resolution = r

        hidden1 = nn.ModuleList()
        hidden2 = nn.ModuleList()
        hidden3 = nn.ModuleList()
        hidden4 = nn.ModuleList()
        hidden5 = nn.ModuleList()
        hidden6 = nn.ModuleList()
        hidden7 = nn.ModuleList()
        hidden8 = nn.ModuleList()
        hidden9 = nn.ModuleList()
        for i in range(math.ceil(DEPTHS[0]*depth)):
            if i == 0:
                hidden1.append(nn.Conv2d(
                    in_channels=in_channels, out_channels=math.ceil(32*width), kernel_size=3, stride=2, padding=1))
                hidden2.append(
                    MBConv(in_channels=math.ceil(32*width), out_channels=math.ceil(16*width), t=1, kernel_size=3, padding=3//2))
                hidden8.append(
                    MBConv(in_channels=math.ceil(192*width), out_channels=math.ceil(320*width), t=6, kernel_size=3, padding=3//2))
                hidden9.append(nn.Conv2d(in_channels=math.ceil(
                    320*width), out_channels=math.ceil(1280*width), kernel_size=1))
            else:
                hidden1.append(
                    nn.Conv2d(in_channels=math.ceil(32*width), out_channels=math.ceil(32*width), kernel_size=3, padding=3//2))
                hidden2.append(
                    MBConv(in_channels=math.ceil(16*width), out_channels=math.ceil(16*width), t=1, kernel_size=3, padding=3//2))
                hidden8.append(
                    MBConv(in_channels=math.ceil(320*width), out_channels=math.ceil(320*width), t=6, kernel_size=3, padding=3//2))
                hidden9.append(nn.Conv2d(in_channels=math.ceil(
                    1280*width), out_channels=math.ceil(1280*width), kernel_size=1))

        for i in range(math.ceil(DEPTHS[2]*depth)):
            if i == 0:
                hidden3.append(
                    MBConv(in_channels=math.ceil(16*width), out_channels=math.ceil(24*width), t=6, kernel_size=3, stride=2))
                hidden4.append(MBConv(
                    in_channels=math.ceil(24*width), out_channels=math.ceil(40*width), t=6, kernel_size=5, stride=2, padding=2))
            else:
                hidden3.append(
                    MBConv(in_channels=math.ceil(24*width), out_channels=math.ceil(24*width), t=6, kernel_size=3, padding=3//2))
                hidden4.append(MBConv(
                    in_channels=math.ceil(40*width), out_channels=math.ceil(40*width), t=6, kernel_size=5, padding=5//2))

        for i in range(math.ceil(DEPTHS[4]*depth)):
            if i == 0:
                hidden5.append(
                    MBConv(in_channels=math.ceil(40*width), out_channels=math.ceil(80*width), t=6, kernel_size=3, padding=3//2))
                hidden6.append(MBConv(
                    in_channels=math.ceil(80*width), out_channels=math.ceil(112*width), t=6, kernel_size=5, stride=2, padding=2))
            else:
                hidden5.append(
                    MBConv(in_channels=math.ceil(80*width), out_channels=math.ceil(80*width), t=6, kernel_size=3, padding=3//2))
                hidden6.append(MBConv(
                    in_channels=math.ceil(112*width), out_channels=math.ceil(112*width), t=6, kernel_size=5, padding=5//2))

        for i in range(math.ceil(DEPTHS[6]*depth)):
            if i == 0:
                hidden7.append(MBConv(
                    in_channels=math.ceil(112*width), out_channels=math.ceil(192*width), t=6, kernel_size=5, stride=2, padding=2))
            else:
                hidden7.append(MBConv(
                    in_channels=math.ceil(192*width), out_channels=math.ceil(192*width), t=6, kernel_size=5, padding=5//2))

        self.efficientNet = nn.ModuleList()
        self.efficientNet.extend(iter(
            [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7, hidden8, hidden9]))

        self.out_9 = nn.Sequential(
            nn.Conv2d(in_channels=math.ceil(1280*width),
                      out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

        self.out_6 = nn.Sequential(
            nn.Conv2d(in_channels=math.ceil(112*width),
                      out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

        self.out_4 = nn.Sequential(
            nn.Conv2d(in_channels=math.ceil(40*width),
                      out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

        self.out_3 = nn.Sequential(
            nn.Conv2d(in_channels=math.ceil(24*width),
                      out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):

        x = F.interpolate(x, size=self.resolution)
        out_9 = None
        out_6 = None
        out_4 = None
        out_3 = None
        for i, hidden in enumerate(self.efficientNet):
            for _, layer in enumerate(hidden):
                x = layer(x)

            if i == 8:
                out_9 = F.leaky_relu(self.out_9(x))
            if i == 5:
                out_6 = F.leaky_relu(self.out_6(x))
            if i == 3:
                out_4 = F.leaky_relu(self.out_4(x))
            if i == 2:
                out_3 = F.leaky_relu(self.out_3(x))

        return x, out_3, out_4, out_6, out_9


def EfficientNetB0(original_aspect_ratio):
    resol = (224, math.ceil(224/original_aspect_ratio))
    print("efficient_net input resolution: ", resol)
    return efficient_net(3, 1, 1, resol)


def EfficientNetB1(original_aspect_ratio):
    resol = (240, math.ceil(240/original_aspect_ratio))
    print("efficient_net input resolution: ", resol)
    return efficient_net(3, 1.0, 1.1, resol)


def EfficientNetB2(original_aspect_ratio):
    resol = (260, math.ceil(260/original_aspect_ratio))
    print("efficient_net input resolution: ", resol)
    return efficient_net(3, 1.1, 1.2, resol)


def EfficientNetB3(original_aspect_ratio):
    resol = (300, math.ceil(300/original_aspect_ratio))
    print("efficient_net input resolution: ", resol)
    return efficient_net(3, 1.2, 1.4, resol)


def EfficientNetB4(original_aspect_ratio):
    resol = (380, math.ceil(380/original_aspect_ratio))
    print("efficient_net input resolution: ", resol)
    return efficient_net(3, 1.4, 1.8, resol)


def EfficientNetB5(original_aspect_ratio):
    resol = (456, math.ceil(456/original_aspect_ratio))
    print("resol", resol)
    return efficient_net(3, 1.6, 2.2, resol)


def EfficientNetB6(original_aspect_ratio):
    resol = (528, math.ceil(528/original_aspect_ratio))
    print("efficient_net input resolution: ", resol)
    return efficient_net(3, 1.8, 2.6, resol)


def EfficientNetB7(original_aspect_ratio):
    resol = (600, math.ceil(600/original_aspect_ratio))
    print("efficient_net input resolution: ", resol)
    return efficient_net(3, 2.0, 3.1, resol)
