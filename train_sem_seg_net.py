# %%
import os.path
import sys
from ignite.engine import Events, Engine
import numpy as np
# # from ignite.contrib.handlers.param_scheduler import PiecewiseLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from utils.get_vkitti_dataset_full import get_dataloaders

from utils.tensorize_batch import tensorize_batch
from uitils.convert_tensor_to_RGB import convert_tensor_to_RGB

from utils.get_stuff_thing_classes import get_stuff_thing_classes
from utils.data_loader_2_coco_ann import data_loader_2_coco_ann
from eval_sem_seg import eval_sem_seg

from torch.utils.tensorboard import SummaryWriter


import config_kitti
import temp_variables
import constants
import models

from datetime import datetime
# from utils import map_hasty
# from utils import get_splits
# import matplotlib.pyplot as plt


# %%


writer = SummaryWriter()




def __update_model(trainer_engine, batch):
    model.train()
    optimizer.zero_grad()

    imgs, ann, _, _, _, _, _, _, _ = batch

    imgs = list(img for img in imgs)
    imgs = tensorize_batch(imgs, device)

    annotations = [{k: v.to(device) for k, v in t.items()}
                   for t in ann]
    
    # print("shape", imgs.shape, sparse_depth.shape, sparse_depth_gt.shape)
    loss_dict = model(imgs, anns=annotations)

    losses = sum(loss for loss in loss_dict.values())

    i = trainer_engine.state.iteration
    writer.add_scalar("Loss/train/iteration", losses, i)

    for key in loss_dict.keys():
        writer.add_scalar("Loss/train/{}".format(key), loss_dict[key], i)

    losses.backward()

    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    optimizer.step()

    return losses


# %% Define Event listeners


def __log_training_loss(trainer_engine):

    current_lr = None

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']

    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    i = trainer_engine.state.iteration
    text = "Epoch {}/{} : {} - batch loss: {:.2f}, LR:{}".format(
        state_epoch, max_epochs, i, batch_loss, current_lr)

    sys.stdout = open(train_res_file, 'a+')
    print(text)

    # writer.add_scalar("LR/train/iteration", current_lr, i)


def __log_validation_results(trainer_engine):
    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    weights_path = "{}{}_loss_{}.pth".format(
        constants.MODELS_LOC, config_kitti.MODEL, batch_loss)
    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)

    sys.stdout = open(train_res_file, 'a+')
    print("Model weights filename: ", weights_path)
    text = "Validation Results - Epoch {}/{} batch_loss: {:.2f}".format(
        state_epoch, max_epochs, batch_loss)
    sys.stdout = open(train_res_file, 'a+')
    print(text)

    miou, rgb_sample, mask_gt, mask_output = eval_sem_seg(model, data_loader_val, weights_path)

    mask_gt = convert_tensor_to_RGB(mask_gt.unsqueeze(0)).squeeze(0)
    mask_output = convert_tensor_to_RGB(mask_output.unsqueeze(0)).squeeze(0)

    writer.add_scalar("mIoU/train/epoch", miou, state_epoch)
    writer.add_image("eval/src_img", rgb_sample, state_epoch, dataformats="CHW")
    writer.add_image("eval/gt", mask_gt, state_epoch, dataformats="CHW")
    writer.add_image("eval/out", mask_output, state_epoch, dataformats="CHW")

    
    scheduler.step()


if __name__ == "__main__":

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    res_filename = "results_{}_{}".format(config_kitti.MODEL, timestamp)
    train_res_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.RES_LOC, res_filename)

    with open(train_res_file, "w+") as training_results:
        training_results.write(
            "----- TRAINING RESULTS - Vkitti {} ----".format(config_kitti.MODEL)+"\n")
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)
    temp_variables.DEVICE = device
    # Empty cuda cache
    torch.cuda.empty_cache()

    # Get model according to config
    model = models.get_model_by_name(config_kitti.MODEL)
    # move model to the right device
    model.to(device)

    print("Allocated memory: ", torch.cuda.memory_allocated(device=device))
    # Define params
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(
    #     params, lr=0.005, momentum=0.9, weight_decay=0.00005)

    optimizer = torch.optim.SGD(
        params, lr=0.0016, momentum=0.9, weight_decay=0.00005)

#     # Set Categories
    ann_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), config_kitti.COCO_ANN)
    all_categories, stuff_categories, thing_categories = get_stuff_thing_classes(
        ann_file)



    data_loader_train = None
    data_loader_val = None

    if config_kitti.USE_PREEXISTING_DATA_LOADERS:
        data_loader_train = torch.load(config_kitti.DATA_LOADER_TRAIN_FILANME)
        data_loader_val = torch.load(config_kitti.DATA_LOADER_VAL_FILENAME)

        # Dataloader to coco ann for evaluation purposes
        annotation = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.COCO_ANN)
        data_loader_2_coco_ann(config_kitti.DATA_LOADER_VAL_FILENAME, annotation)

    else:

        imgs_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "vkitti_2.0.3_rgb/")

        depth_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "vkitti_2.0.3_depth/")

        annotation = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.COCO_ANN)

        data_loader_train, data_loader_val = get_dataloaders(
            config_kitti.BATCH_SIZE,
            imgs_root,
            depth_root,
            annotation,
            split=True,
            val_size=config_kitti.VAL_SIZE,
            n_samples=config_kitti.MAX_TRAINING_SAMPLES)

        # save data loaders
        data_loader_train_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.VKITTI_DATA_LOADER_TRAIN_FILANME)

        data_loader_val_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.VKITTI_DATA_LOADER_VAL_FILENAME)

        torch.save(data_loader_train, data_loader_train_filename)
        torch.save(data_loader_val, data_loader_val_filename)

        # Dataloader to coco ann for evaluation purposes
        data_loader_2_coco_ann(data_loader_val_filename, annotation)

    # ---------------TRAIN--------------------------------------
    scheduler = MultiStepLR(optimizer, milestones=[65, 80, 85, 90], gamma=0.1)
    ignite_engine = Engine(__update_model)

    # ignite_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
    ignite_engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=50), __log_training_loss)
    ignite_engine.add_event_handler(
        Events.EPOCH_COMPLETED, __log_validation_results)
    ignite_engine.run(data_loader_train, config_kitti.MAX_EPOCHS)
    writer.flush()
