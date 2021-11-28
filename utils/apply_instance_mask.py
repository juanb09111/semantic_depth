
import torch 
import random

def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()
    g = random.random()
    b = random.random()
    rgb = [r, g, b]
    return rgb

def apply_instance_masks(image, masks, confidence, device, ids=None):

    masks = masks.squeeze(1)
    
    background_mask = torch.zeros((1, *masks[0].shape), device=device)
    background_mask = background_mask.new_full(background_mask.shape, confidence)
    
    masks = torch.cat([background_mask, masks], dim=0)
    
    mask_argmax = torch.argmax(masks, dim=0)
    
    if ids is not None:
        mask = torch.tensor(background_mask, dtype=torch.long, device=device)
        for idx, obj_id in enumerate(ids):
            mask = torch.where((mask_argmax == idx + 1), obj_id, mask)
    else:
        mask = mask_argmax

    max_val = mask.max()

    for i in range(1, max_val + 1):
        for c in range(3):
            alpha = 0.45
            color = randRGB(i)
            image[c, :, :] = torch.where(mask == i,
                                      image[c, :, :] *
                                      (1 - alpha) + alpha * color[c],
                                      image[c, :, :])
    return image