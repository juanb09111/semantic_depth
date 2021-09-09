import torch
import torch.nn as nn


class_colors = [
    [0, 0, 0],
    [210, 0, 200],
    [90, 200, 255],
    [0 ,199, 0],
    [90, 240, 0],
    [140, 140, 140],
    [100, 60, 100],
    [250, 100, 255],
    [255, 255, 0],
    [200, 200, 0],
    [255, 130, 0],
    [80, 80, 80],
    [160, 60, 60],
    [255, 127, 80],
    [0, 139, 139]
]

def convert_tensor_to_RGB(network_output, device):
    x = torch.cuda.FloatTensor(class_colors)
    network_output = torch.tensor(network_output, device=device).to(torch.int64)
    converted_tensor = nn.functional.embedding(network_output, x).permute(0, 3, 1, 2)
    return torch.tensor(converted_tensor, device=device)
