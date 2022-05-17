# %%

import os.path
import math
import torch
# from pycocotools.coco import COCO
from ytvostools.ytvos import YTVOS
from ytvostools import mask
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from os import walk

# %%
torch.manual_seed(0)

class youtubeTestDataset(torch.utils.data.Dataset):

    def __init__(self,
                 imgs_root,
                 transforms=None):

        self.transforms = transforms
        self.imgs_root = imgs_root

        self.frames = next(walk(self.imgs_root), (None, None, []))[2]  # [] if no file
        print(self.frames)

    def __getitem__(self, index):

        frame = self.frames[index]
       
        img_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", self.imgs_root, frame)

        source_img = Image.open(img_filename)

        if self.transforms is not None:
            source_img = self.transforms(source_img)
            #transform annotations too
        
        return source_img
    
    def __len__(self):
        return len(self.frames)

class youtubeDataset(torch.utils.data.Dataset):

    def __init__(self,
                 imgs_root,
                 annotation,
                 transforms=None,
                 n_samples=None,
                 shuffle=True):

        self.transforms = transforms
        self.imgs_root = imgs_root
        self.ytvos = YTVOS(annotation)

        self.vidIds = self.ytvos.getVidIds()
        self.videos = self.ytvos.loadVids(ids=self.vidIds)
        self.frames = []
        for video in self.videos:
            file_names = video["file_names"]
            video_id = video["id"]
            annIds = self.ytvos.getAnnIds(vidIds=[video_id])
            anns = self.ytvos.loadAnns(ids=annIds)

            num_objs = len(anns)
            video_cats = [anns[i]["category_id"] for i in range(num_objs)]
            video_iscrowd = [anns[i]["iscrowd"] for i in range(num_objs)]
           
            for idx, file_name in enumerate(file_names):
                frame_anns = [
                    el for el in
                    [anns[i]["segmentations"][idx] for i in range(num_objs)]
                    if el is not None
                ]
                frame_boxes = [
                    el for el in
                    [anns[i]["bboxes"][idx] for i in range(num_objs)]
                    if el is not None
                ]
                frame_areas = [
                    el
                    for el in [anns[i]["areas"][idx] for i in range(num_objs)]
                    if el is not None
                ]

                if len(frame_anns) > 0:
                    frame_cats = [
                        video_cats[j] for j in range(len(frame_anns))
                    ]
                    frame_iscrowd = [
                        video_iscrowd[j] for j in range(len(frame_anns))
                    ]

                    self.frames.append({
                        "file_name": file_name,
                        "anns": frame_anns,
                        "frame_cats": frame_cats,
                        "bboxes": frame_boxes,
                        "areas": frame_areas,
                        "iscrowd": frame_iscrowd
                    })

        if shuffle:
            print("Shuffling samples")
            random.Random(4).shuffle(self.frames)

        if n_samples != None:
            self.frames = self.frames[:n_samples]

    def __getitem__(self, index):

        frame = self.frames[index]
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]]
                 for box in frame["bboxes"]]
        labels = frame["frame_cats"]
        areas = frame["areas"]
        iscrowd = frame["iscrowd"]
        masks = [
            mask.decode(mask.frPyObjects(rle, rle['size'][0], rle['size'][1]))
            for rle in frame["anns"]
        ]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        annotation = {}
        annotation["boxes"] = boxes
        annotation["labels"] = labels
        annotation["area"] = areas
        annotation["iscrowd"] = iscrowd
        annotation['masks'] = masks

        img_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", self.imgs_root, frame["file_name"])

        source_img = Image.open(img_filename)

        if self.transforms is not None:
            source_img = self.transforms(source_img)
            #transform annotations too
        
        return source_img, annotation
    
    def __len__(self):
        return len(self.frames)


data_transform = transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_datasets(imgs_root, annotation=None, split=False, val_size=0.20, n_samples=None, shuffle=True, is_test_set=False, reverse=False):

    if is_test_set:
        yt_test_dataset = youtubeTestDataset(imgs_root, data_transform)
        return yt_test_dataset

    ytdataset = youtubeDataset(imgs_root, annotation, transforms=data_transform, shuffle=shuffle, n_samples=n_samples)

    if split:
        if val_size >= 1:
            raise AssertionError(
                "val_size must be a value within the range of (0,1)")

        len_val = math.ceil(len(ytdataset)*val_size)
        len_train = len(ytdataset) - len_val

        if len_train < 1 or len_val < 1:
            raise AssertionError("datasets length cannot be zero")

        indices = list(range(len(ytdataset)))
        train_set = torch.utils.data.Subset(ytdataset, indices[:len_train])
        val_set = torch.utils.data.Subset(ytdataset, indices[len_train:])

        return train_set, val_set
    else:
        return ytdataset

def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(batch_size,
                    imgs_root,
                    annotation,
                    num_replicas,
                    rank,
                    split=False,
                    val_size=0.20,
                    n_samples=None,
                    sampler=True,
                    shuffle=True,
                    is_test_set=False,
                    reverse=False):

    if split:
        train_set, val_set = get_datasets(
            imgs_root, annotation, split=True, val_size=val_size, n_samples=n_samples, shuffle=shuffle, is_test_set=is_test_set, reverse=reverse)

        train_sampler = None
        val_sampler = None

        if sampler:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_set, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        num_workers=4,
                                                        collate_fn=collate_fn,
                                                        drop_last=True,
                                                        sampler=train_sampler)

        data_loader_val = torch.utils.data.DataLoader(val_set,
                                                      batch_size=batch_size,
                                                      num_workers=4,
                                                      collate_fn=collate_fn,
                                                      drop_last=True,
                                                      sampler=val_sampler)
        return data_loader_train, data_loader_val

    else:
        dataset = get_datasets(imgs_root, annotation, split=False, val_size=val_size, n_samples=n_samples, shuffle=shuffle, is_test_set=is_test_set, reverse=reverse)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  num_workers=4,
                                                  collate_fn=collate_fn,
                                                  drop_last=True)
    return data_loader


# imgs_root = "datasets/youtube_vis_2019/train/JPEGImages"
# annotation = "datasets/youtube_vis_2019/train.json"

# ytdataset = youtubeDataset(imgs_root, annotation, transforms=data_transform)
# ytdataset.__getitem__(0)
