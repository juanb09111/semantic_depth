import os.path
import sys
from ignite.engine import Events, Engine
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR
from utils.get_vkitti_dataset_full import get_dataloaders

from utils.tensorize_batch import tensorize_batch
from utils.convert_tensor_to_RGB import convert_tensor_to_RGB
from utils.data_loader_2_coco_ann import data_loader_2_coco_ann
from utils.get_stuff_thing_classes import get_stuff_thing_classes

from eval_scripts.eval_panoptic_depth import eval_panoptic_depth

from torch.utils.tensorboard import SummaryWriter


import config_kitti
import temp_variables
import constants
import models

from datetime import datetime


# %%



def __update_model_wrapper(model, optimizer, device, rank, writer):
    def __update_model(trainer_engine, batch):
        model.train()
        optimizer.zero_grad()

        imgs, ann, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, _, _ = batch

        imgs = list(img for img in imgs)
        lidar_fov = list(lid_fov for lid_fov in lidar_fov)
        masks = list(mask for mask in masks)
        sparse_depth = list(sd for sd in sparse_depth)
        k_nn_indices = list(k_nn for k_nn in k_nn_indices)
        sparse_depth_gt = list(sp_d for sp_d in sparse_depth_gt)

        imgs = tensorize_batch(imgs, device)
        lidar_fov = tensorize_batch(lidar_fov, device, dtype=torch.float)
        masks = tensorize_batch(masks, device, dtype=torch.bool)
        sparse_depth = tensorize_batch(sparse_depth, device)
        k_nn_indices = tensorize_batch(k_nn_indices, device, dtype=torch.long)
        sparse_depth_gt = tensorize_batch(
            sparse_depth_gt, device, dtype=torch.float)

        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in ann]

       

        # print("shape", imgs.shape, sparse_depth.shape, sparse_depth_gt.shape)
        loss_dict = model(imgs,
                          sparse_depth,
                          masks,
                          lidar_fov,
                          k_nn_indices,
                          sparse_depth_gt=sparse_depth_gt,
                          anns=annotations)

        # losses = sum(loss for loss in loss_dict.values())

        i = trainer_engine.state.iteration
        epoch = trainer_engine.state.epoch

        if epoch <= 50:
            losses = sum(loss[1] for loss in loss_dict.items() if loss[0] is "depth_loss")

        if epoch <= 100 and i > 50:
            losses = sum(loss[1] for loss in loss_dict.items() if loss[0] is "semantic_loss")

        if epoch <= 150 and i > 100:
            losses = sum(loss[1] for loss in loss_dict.items() if loss[0] not in ["semantic_loss", "depth_loss"] )

        if epoch <= 200 and i > 150:
            losses = sum(loss for loss in loss_dict.values())

        if rank == 0:
            # print(losses)
            writer.add_scalar("Loss/train/iteration", losses, i)

            for key in loss_dict.keys():
                writer.add_scalar(
                    "Loss/train/{}".format(key), loss_dict[key], i)

        losses.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        optimizer.step()

        return losses
    return __update_model


# %% Define Event listeners

def __log_training_loss_wrapper(optimizer, train_res_file):
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
    return __log_training_loss


def __log_validation_results_wrapper(model, optimizer, data_loader_val, all_categories, thing_categories, scheduler, rank, train_res_file, device, writer):
    def __log_validation_results(trainer_engine):
        batch_loss = trainer_engine.state.output
        state_epoch = trainer_engine.state.epoch
        max_epochs = trainer_engine.state.max_epochs
        i = trainer_engine.state.iteration
        weights_path = "{}{}_loss_{}.pth".format(
            constants.MODELS_LOC, "Panoptic_Depth", batch_loss)

        if rank == 0:
            dict_model = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': state_epoch,
                'iteration': i
            }
            torch.save(dict_model, weights_path)

        sys.stdout = open(train_res_file, 'a+')
        print("Model weights filename: ", weights_path)
        text = "Validation Results - Epoch {}/{} batch_loss: {:.2f}".format(
            state_epoch, max_epochs, batch_loss)
        sys.stdout = open(train_res_file, 'a+')
        print(text)

        if rank == 0:
            rmse, miou, rgb_sample, mask_gt, mask_output, sparse_depth_gt_sample, sparse_depth_gt_full, out_depth = eval_panoptic_depth(
                model, data_loader_val, weights_path, all_categories, thing_categories, device)

            mask_gt = convert_tensor_to_RGB(mask_gt.unsqueeze(0), device).squeeze(0)/255
            mask_output = torch.argmax(mask_output, dim=0)
            mask_output = convert_tensor_to_RGB(mask_output.unsqueeze(0), device).squeeze(0)/255

            writer.add_scalar("Loss/train/epoch", batch_loss, state_epoch)
            writer.add_scalar("rmse/train/epoch", rmse, state_epoch)
            writer.add_scalar("mIoU/train/epoch", miou, state_epoch)

            writer.add_image("eval/src_img", rgb_sample, state_epoch, dataformats="CHW")
            writer.add_image("eval/gt", mask_gt, state_epoch, dataformats="CHW")
            writer.add_image("eval/out", mask_output, state_epoch, dataformats="CHW")


            # Depth
            sparse_depth_gt_sample = sparse_depth_gt_sample.squeeze_(0)
            sparse_depth_gt_sample = sparse_depth_gt_sample.cpu().numpy()/255
            out_depth_numpy = out_depth.cpu().numpy()/255
            sparse_depth_gt_full = sparse_depth_gt_full.cpu().numpy()/255

            writer.add_image("eval_depth/src_img", rgb_sample,
                            state_epoch, dataformats="CHW")
            writer.add_image("eval_depth/gt_full", sparse_depth_gt_full,
                            state_epoch, dataformats="HW")
            writer.add_image("eval_depth/gt", sparse_depth_gt_sample,
                            state_epoch, dataformats="HW")
            writer.add_image("eval_depth/out", out_depth_numpy,
                            state_epoch, dataformats="HW")

        scheduler.step()
    return __log_validation_results


def __setup_state_wrapper(start_epoch, start_iteration):
    def __setup_state(engine):
        engine.state.epoch = start_epoch
        engine.state.iteration = start_iteration
    return __setup_state


def train(gpu, args):
    # DP
    args.gpu = gpu
    print('gpu:', gpu)
    # rank calculation for each process per gpu so that they can be identified uniquely.
    # rank = int(os.environ.get("SLURM_NODEID")) * args.ngpus + gpu
    rank = args.local_ranks * args.ngpus + gpu
    print('rank:', rank)

    torch.manual_seed(0)
    # start from the same randomness in different nodes. If you don't set it
    # then networks can have different weights in different nodes when the
    # training starts. We want exact copy of same network in all the nodes.
    # Then it will progress from there.

    # set the gpu for each processes
    torch.cuda.set_device(args.gpu)

    # Write results in text file

    res_filename = "results_{}".format("PanopticDepth")
    train_res_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", constants.RES_LOC, res_filename)

    with open(train_res_file, "w+") as training_results:
        training_results.write(
            "----- TRAINING RESULTS - Vkitti {} ----".format("PanopticDepth")+"\n")
    # Set device
    temp_variables.DEVICE = args.gpu

    print("Device: ", args.gpu)
    # Empty cuda cache
    torch.cuda.empty_cache()

    # Get model according to config
    model = models.get_model_by_name("PanopticDepth").cuda(args.gpu)

    # move model to the right device

    print("Allocated memory: ", torch.cuda.memory_allocated(device=args.gpu))
    # Define params
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(
    #     params, lr=0.005, momentum=0.9, weight_decay=0.00005)

    # fusenet/semantic optimizer
    # optimizer = torch.optim.SGD(
    #     params, lr=0.0016, momentum=0.9, weight_decay=0.00005)

    # instance
    # optimizer = torch.optim.SGD(
        # params, lr=0.0025, momentum=0.9, weight_decay=0.0005)

    # Panoptic depth optimizer
    # optimizer = torch.optim.SGD(
    #     params, lr=0.0001, momentum=0.9, weight_decay=0.00005)

    # joint 
    optimizer = torch.optim.SGD(
        params, lr=0.0016, momentum=0.9, weight_decay=0.00005)

    if args.checkpoint is not None:

        sys.stdout = open(train_res_file, 'a+')
        print("Loading checkpoint from {} to {}".format(0, rank), args.checkpoint)

        checkpoint = torch.load(args.checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(args.gpu)

    data_loader_train = None
    data_loader_val = None

    annotation = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", config_kitti.COCO_ANN)

    all_categories, _, thing_categories = get_stuff_thing_classes(annotation)

    if config_kitti.USE_PREEXISTING_DATA_LOADERS:
        data_loader_train = torch.load(config_kitti.DATA_LOADER_TRAIN_FILANME)
        data_loader_val = torch.load(config_kitti.DATA_LOADER_VAL_FILENAME)

        # # Dataloader to coco ann for evaluation purposes

        data_loader_2_coco_ann(
            config_kitti.DATA_LOADER_VAL_FILENAME, annotation)

    else:

        imgs_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "..", config_kitti.DATA, "vkitti_2.0.3_rgb/")
        
        semantic_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "..", config_kitti.DATA, "semseg_bin/") 

        depth_root = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "..", config_kitti.DATA, "vkitti_2.0.3_depth/")

        data_loader_train, data_loader_val = get_dataloaders(
            args.batch_size,
            imgs_root,
            semantic_root,
            depth_root,
            annotation,
            num_replicas=args.world_size,
            rank=rank,
            split=True,
            val_size=config_kitti.VAL_SIZE,
            n_samples=config_kitti.MAX_TRAINING_SAMPLES,
            sampler=False,
            shuffle=True,
            is_test_set=False)

        # save data loaders
        data_loader_train_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", constants.DATA_LOADERS_LOC, constants.VKITTI_DATA_LOADER_TRAIN_FILANME)

        data_loader_val_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", constants.DATA_LOADERS_LOC, constants.VKITTI_DATA_LOADER_VAL_FILENAME)

        torch.save(data_loader_train, data_loader_train_filename)
        torch.save(data_loader_val, data_loader_val_filename)

        # Dataloader to coco ann for evaluation purposes
        data_loader_2_coco_ann(data_loader_val_filename, annotation)

    if rank == 0:
        writer = SummaryWriter(log_dir="runs/PanopticDepth_2")
    else:
        writer = None

    # ---------------TRAIN--------------------------------------
    scheduler = MultiStepLR(optimizer, milestones=[150, 185], gamma=0.1)
    ignite_engine = Engine(__update_model_wrapper(
        model, optimizer, args.gpu, rank, writer))

    if args.checkpoint is not None:
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        ignite_engine.add_event_handler(
            Events.STARTED, __setup_state_wrapper(epoch, iteration))

    ignite_engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=20), __log_training_loss_wrapper(optimizer, train_res_file))
    ignite_engine.add_event_handler(
        Events.EPOCH_COMPLETED, __log_validation_results_wrapper(model,
                                                                 optimizer,
                                                                 data_loader_val,
                                                                 all_categories,
                                                                 thing_categories,
                                                                 scheduler,
                                                                 rank,
                                                                 train_res_file,
                                                                 gpu, writer))
    ignite_engine.run(data_loader_train, config_kitti.MAX_EPOCHS)

    if rank == 0:
        writer.flush()