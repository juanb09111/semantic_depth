# %%
import os.path
import sys
from ignite.engine import Events, Engine

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR
from utils.get_youtube_dataloader import get_dataloaders

from utils.get_stuff_thing_classes import get_stuff_thing_classes

from torch.utils.tensorboard import SummaryWriter


import temp_variables
import constants
import models

from datetime import datetime



# %%





def __update_model_wrapper(model, optimizer, device, rank, writer):
    def __update_model(trainer_engine, batch):
        model.train()
        optimizer.zero_grad()

        # imgs, ann, _, _, _, _, _, _, _ = batch
        imgs, ann = batch

        imgs = list(img for img in imgs)
        imgs = [img.to(device) for img in imgs]

        annotations = [{k: v.to(device) for k, v in t.items()}
                    for t in ann]
        
        # print("shape", imgs.shape, sparse_depth.shape, sparse_depth_gt.shape)
        loss_dict = model(imgs, anns=annotations)

        losses = sum(loss for loss in loss_dict.values())

        i = trainer_engine.state.iteration
        if rank==0:
            writer.add_scalar("Loss/train/iteration", losses, i)

            for key in loss_dict.keys():
                writer.add_scalar("Loss/train/{}".format(key), loss_dict[key], i)

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


def __log_validation_results_wrapper(model, optimizer, all_categories, thing_categories, scheduler, rank, train_res_file, device, writer):
    def __log_validation_results(trainer_engine):
        batch_loss = trainer_engine.state.output
        state_epoch = trainer_engine.state.epoch
        max_epochs = trainer_engine.state.max_epochs
        i = trainer_engine.state.iteration
        weights_path = "{}{}_loss_{}.pth".format(
            constants.MODELS_LOC, "MaskRcnn", batch_loss)
        
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
    
    res_filename = "results_{}_lr={}".format("MaskRcnn", args.lr)
    train_res_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", constants.RES_LOC, res_filename)

    with open(train_res_file, "w+") as training_results:
        training_results.write(
            "----- TRAINING RESULTS  {} ----".format("MaskRcnn")+"\n")
    # Set device
    temp_variables.DEVICE = args.gpu
    
    print("Device: ", args.gpu)
    # Empty cuda cache
    torch.cuda.empty_cache()

    # Get model according to config
    model = models.get_model_by_name("MaskRcnn").cuda(args.gpu)

        

    # move model to the right device

    print("Allocated memory: ", torch.cuda.memory_allocated(device=args.gpu))
    # Define params
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(
    #     params, lr=0.005, momentum=0.9, weight_decay=0.00005)

    # optimizer = torch.optim.SGD(
        # params, lr=0.0002, momentum=0.9, weight_decay=0.00005)

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
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

    
    #TODO: map location utils.maplocation

    # annotation = os.path.join(os.path.dirname(os.path.abspath(
    #         __file__)), "..", config_kitti.COCO_ANN)
    
    annotation = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "..", args.ann_file)
    
    
    all_categories, _, thing_categories = get_stuff_thing_classes(annotation)

    imgs_root = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", args.data)
    

    data_loader_train = get_dataloaders(
        args.batch_size,
        imgs_root,
        annotation,
        num_replicas=args.world_size,
        rank=rank,
        split=False,
        val_size=args.val_size,
        n_samples=args.n_samples,
        sampler=False,
        shuffle=False,
        is_test_set=False)

    # save data loaders
    data_loader_train_filename = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", constants.DATA_LOADERS_LOC, constants.YTVOS_DATA_LOADER_TRAIN_FILANME)

    torch.save(data_loader_train, data_loader_train_filename)

    if rank ==0:
        writer = SummaryWriter(log_dir="runs/MaskRcnn_ytvos_lr={}_momentum=0.9_weight_decay=0.0005".format(args.lr))
    else:
        writer=None

    # ---------------TRAIN--------------------------------------
    scheduler = MultiStepLR(optimizer, milestones=[65, 80, 85, 90], gamma=0.1)
    ignite_engine = Engine(__update_model_wrapper(model, optimizer, args.gpu, rank, writer))

    if  args.checkpoint is not None:
        epoch = checkpoint['epoch']
        iteration: checkpoint['iteration']
        ignite_engine.add_event_handler(Events.STARTED, __setup_state_wrapper(epoch, iteration))

    ignite_engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=20), __log_training_loss_wrapper(optimizer, train_res_file))
    ignite_engine.add_event_handler(
        Events.EPOCH_COMPLETED, __log_validation_results_wrapper(model,
        optimizer,  
        all_categories, 
        thing_categories, 
        scheduler, 
        rank, 
        train_res_file, 
        gpu, writer))
    ignite_engine.run(data_loader_train, args.epochs)

    if rank==0:
        writer.flush()

