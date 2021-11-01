import torch
import config_kitti
import random


def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()
    g = random.random()
    b = random.random()
    rgb = [r, g, b]
    return rgb


def get_colors_palete(num_classes):
    colors = [randRGB(i+5) for i in range(num_classes + 1)]
    return colors

def apply_panoptic_mask_gpu(image, mask):
    
    num_stuff_classes = config_kitti.NUM_STUFF_CLASSES
    max_val = mask.max()

    colors = get_colors_palete(config_kitti.NUM_THING_CLASSES + config_kitti.NUM_STUFF_CLASSES)

    for i in range(1, max_val + 1):
        for c in range(3):

            if i <= num_stuff_classes:
                color = colors[i]
                alpha = 0.45

            else:
                alpha = 0.45
                color = randRGB(i)
            
            image[c, :, :] = torch.where(mask == i,
                                      image[c, :, :] *
                                      (1 - alpha) + alpha * color[c],
                                      image[c, :, :])
    return image