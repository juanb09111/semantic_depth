# %%
import torch
from torch import nn
import torch.nn.functional as F


class squeeze_excitation(nn.Module):
    def __init__(self, n_features, r=16):
        super().__init__()

        self.linear1 = nn.Linear(
            n_features, n_features // r, bias=True)
        self.linear2 = nn.Linear(
            n_features // r, n_features, bias=True)

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        # Convert to row vectors
        y = y.permute(0, 2, 3, 1)
        y = F.relu(self.linear1(y))
        y = F.sigmoid(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

# %%
