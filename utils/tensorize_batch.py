import torch

def tensorize_batch(batch, device, dtype=None):
    batch_size= len(batch)
    sample = batch[0]

    if dtype:
        res = torch.zeros((batch_size, *sample.shape), dtype=dtype)
    else:
        res = torch.zeros(batch_size, *sample.shape)

    for i in range(batch_size):
        res[i] = batch[i].to(device)

    return res.to(device)