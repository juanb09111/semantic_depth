import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.depth_wise_conv import depth_wise_conv
#%%
class DPC(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise_conv=True):
        super().__init__()

        self.depthwise_conv = depthwise_conv

        if depthwise_conv:
            print("using depthwise conv in DPC module")
            # depthwise conv
            self.init_depthwise = nn.Sequential(
                depth_wise_conv(in_channels, kernel_size=3, padding=((3//2),(3//2)*6), dilation=(1,6)),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_depthwise_conv_1_1 = nn.Sequential(
                depth_wise_conv(in_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_depthwise_conv_6_21 = nn.Sequential(
                depth_wise_conv(in_channels, kernel_size=3, padding=((3//2)*6,(3//2)*21), dilation=(6,21)),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_depthwise_conv_18_15 = nn.Sequential(
                depth_wise_conv(in_channels, kernel_size=3, padding=((3//2)*18,(3//2)*15), dilation=(18,15)),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_depthwise_conv_6_3 = nn.Sequential(
                depth_wise_conv(in_channels, kernel_size=3, padding=((3//2)*6,(3//2)*3), dilation=(6,3)),
                nn.BatchNorm2d(in_channels)
            )
        else:
            print("using traditional conv in DPC module")
            # Traditional conv
            self.init = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=((3//2),(3//2)*6), dilation=(1,6)),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_conv_1_1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3//2),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_conv_6_21 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=((3//2)*6,(3//2)*21), dilation=(6,21)),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_conv_18_15 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=((3//2)*18,(3//2)*15), dilation=(18,15)),
                nn.BatchNorm2d(in_channels)
            )

            self.dialated_conv_6_3 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=((3//2)*6,(3//2)*3), dilation=(6,3)),
                nn.BatchNorm2d(in_channels)
            )

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels*5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        if self.depthwise_conv:
            init = F.leaky_relu(self.init_depthwise(x))
            dialated_conv_1_1 = F.leaky_relu(self.dialated_depthwise_conv_1_1(init))
            dialated_conv_6_21 = F.leaky_relu(self.dialated_depthwise_conv_6_21(init))
            dialated_conv_18_15 = F.leaky_relu(self.dialated_depthwise_conv_18_15(init))
            dialated_conv_6_3 = F.leaky_relu(self.dialated_depthwise_conv_6_3(dialated_conv_18_15))

        else:
            init = F.leaky_relu(self.init(x))
            dialated_conv_1_1 = F.leaky_relu(self.dialated_conv_1_1(init))
            dialated_conv_6_21 = F.leaky_relu(self.dialated_conv_6_21(init))
            dialated_conv_18_15 = F.leaky_relu(self.dialated_conv_18_15(init))
            dialated_conv_6_3 = F.leaky_relu(self.dialated_conv_6_3(dialated_conv_18_15))

        cat = torch.cat((init, dialated_conv_1_1, dialated_conv_6_21, dialated_conv_18_15, dialated_conv_6_3), dim=1)
        
        x = F.leaky_relu(self.out_conv(cat))
        return x

# images = torch.rand((2, 256, 32, 64))
# model = DPC(256, 128)
# print(model)
# x = model(images)
# print(x.shape)