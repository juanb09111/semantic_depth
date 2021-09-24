# %%

import os.path
import math
import torch
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import config_kitti
import glob
import random
import matplotlib.pyplot as plt
# %%
torch.manual_seed(0)


def get_vkitti_files(dirName, ext):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)

    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_vkitti_files(fullPath, ext)
        elif fullPath.find("morning") != -1 and fullPath.find("Camera_0") != -1 and fullPath.find(ext) != -1:
            allFiles.append(fullPath)

    return allFiles


class vkittiDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_root, depth_root, annotation, transforms, n_samples=None, shuffle=False):

        self.imgs_root = imgs_root
        self.depth_root = depth_root

        self.coco = COCO(annotation)

        # get ids and shuffle
        self.ids = list(sorted(self.coco.imgs.keys()))
        if shuffle:
            random.shuffle(self.ids)

        catIds = self.coco.getCatIds()
        categories = self.coco.loadCats(catIds)
        self.categories = list(map(lambda x: x['name'], categories))
        self.bg_categories_ids = self.coco.getCatIds(supNms="background")
        bg_categories = self.coco.loadCats(self.bg_categories_ids)
        self.bg_categories = list(map(lambda x: x['name'], bg_categories))

        self.obj_categories_ids = self.coco.getCatIds(supNms="object")
        obj_categories = self.coco.loadCats(self.obj_categories_ids)
        self.obj_categories = list(map(lambda x: x['name'], obj_categories))

        depth_files = get_vkitti_files(depth_root, "png")
        self.depth_imgs = depth_files

        self.transforms = transforms

        if n_samples != None:
            self.ids = self.ids[:n_samples]

        print("Training/evaluating on {} samples".format(len(self.ids)))

    def find_k_nearest(self, lidar_fov):
        k_number = config_kitti.K_NUMBER
        b_lidar_fov = torch.unsqueeze(lidar_fov, dim=0)

        distances = torch.cdist(b_lidar_fov, b_lidar_fov, p=2)
        _, indices = torch.topk(distances, k_number + 1, dim=2, largest=False)
        indices = indices[:, :, 1:]  # B x N x 3

        return indices.squeeze_(0).long()

    def sample_depth_img(self, depth_tensor):
        (img_height, img_width) = depth_tensor.shape[1:]

        rand_x_coors = []
        rand_y_coors = []

        for i in range(0, config_kitti.N_NUMBER*3):
            rand_x_coors.append(random.randint(0, img_width - 1))

        for k in range(0, config_kitti.N_NUMBER*3):
            rand_y_coors.append(random.randint(0, img_height - 1))

        coors = torch.zeros((config_kitti.N_NUMBER*3, 2))

        # coors in the form of NxHxW
        coors[:, 1] = torch.tensor(rand_x_coors, dtype=torch.long)
        coors[:, 0] = torch.tensor(rand_y_coors, dtype=torch.long)
        coors = torch.tensor(coors, dtype=torch.long)

        # find unique coordinates
        _, indices = torch.unique(coors[:, :2], dim=0, return_inverse=True)
        unique_indices = torch.zeros_like(torch.unique(indices))

        current_pos = 0
        for i, val in enumerate(indices):
            if val not in indices[:i]:
                unique_indices[current_pos] = i
                current_pos += 1

        imPts = coors[unique_indices]

        depth = depth_tensor[0, imPts[:, 0], imPts[:, 1]]/256

        # filter out long ranges of depth
        inds = depth < config_kitti.MAX_DEPTH

        # fig = plt.figure(4)
        # # ax = plt.axes(projection="3d")
        # ax = plt.axes(projection='3d')

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # x_data = imPts[inds, 1]
        # y_data = imPts[inds, 0]
        # z_data = depth[inds]
        # ax.scatter3D(x_data, y_data, z_data, cmap='Greens', s=1)
        # plt.show()
        # imPts in NxHxW
        return imPts[inds, :][:config_kitti.N_NUMBER], depth[inds][:config_kitti.N_NUMBER]

    def sample_depth_gt_img(self, depth_tensor):
        (img_height, img_width) = depth_tensor.shape[1:]

        rand_x_coors = []
        rand_y_coors = []

        for i in range(0, config_kitti.N_NUMBER*3*10):
            rand_x_coors.append(random.randint(0, img_width - 1))

        for k in range(0, config_kitti.N_NUMBER*3*10):
            rand_y_coors.append(random.randint(0, img_height - 1))

        coors = torch.zeros((config_kitti.N_NUMBER*3*10, 2))

        # coors in the form of NxHxW
        coors[:, 1] = torch.tensor(rand_x_coors, dtype=torch.long)
        coors[:, 0] = torch.tensor(rand_y_coors, dtype=torch.long)
        coors = torch.tensor(coors, dtype=torch.long)

        # find unique coordinates
        _, indices = torch.unique(coors[:, :2], dim=0, return_inverse=True)
        unique_indices = torch.zeros_like(torch.unique(indices))

        current_pos = 0
        for i, val in enumerate(indices):
            if val not in indices[:i]:
                unique_indices[current_pos] = i
                current_pos += 1

        imPts = coors[unique_indices]

        depth = depth_tensor[0, imPts[:, 0], imPts[:, 1]]/256

        # filter out long ranges of depth
        inds = depth < config_kitti.MAX_DEPTH

        # fig = plt.figure(4)
        # # ax = plt.axes(projection="3d")
        # ax = plt.axes(projection='3d')

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # x_data = imPts[inds, 1]
        # y_data = imPts[inds, 0]
        # z_data = depth[inds]
        # ax.scatter3D(x_data, y_data, z_data, cmap='Greens', s=1)
        # plt.show()
        # imPts in NxHxW
        return imPts[inds, :][:config_kitti.N_NUMBER], depth[inds][:config_kitti.N_NUMBER]

    def get_coco_ann(self, index):

        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get object annotations ids from coco
        obj_ann_ids = coco.getAnnIds(
            imgIds=img_id, catIds=self.obj_categories_ids)
        # Dictionary: target coco_annotation file for an image containing only object classes
        coco_annotation = coco.loadAnns(obj_ann_ids)
        # path for input image
        img_filename = coco.loadImgs(img_id)[0]['loc']
        # open the input image
        # img = Image.open(path)

        semantic_mask_path = coco.loadImgs(img_id)[0]['semseg_img_filename']
        # create semantic mask

        semantic_mask_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", config_kitti.DATA, semantic_mask_path)
        semantic_mask = Image.open(semantic_mask_path)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks = []
        category_ids = []
        for i in range(num_objs):

            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            category_id = coco_annotation[i]['category_id']
            label = coco.cats[category_id]['name']
            labels.append(self.obj_categories.index(label) + 1)
            area = coco_annotation[i]['area']
            areas.append(area)

            iscrowd.append(coco_annotation[i]['iscrowd'])

            mask = coco.annToMask(coco_annotation[i])
            masks.append(mask)

            category_ids.append(category_id)

        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            areas = torch.as_tensor(
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
            labels = torch.zeros((1), dtype=torch.int64)
            masks = torch.zeros(
                (1, *config_kitti.CROP_OUTPUT_SIZE), dtype=torch.uint8)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Iscrowd

        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

        # Num of instance objects
        num_objs = torch.as_tensor(num_objs, dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["category_ids"] = category_ids
        my_annotation["num_instances"] = num_objs
        my_annotation['masks'] = masks

        if self.transforms is not None:
            semantic_mask = self.transforms(crop=True)(semantic_mask)*255
            semantic_mask = torch.as_tensor(
                semantic_mask, dtype=torch.uint8).squeeze_(0)

        my_annotation["semantic_mask"] = semantic_mask

        return img_filename, my_annotation

    def __getitem__(self, index):

        img_filename, ann = self.get_coco_ann(index)

        scene = img_filename.split("/")[-6]

        basename = img_filename.split(".")[-2].split("_")[-1]

        depth_filename = [s for s in self.depth_imgs if (
            scene in s and basename in s)][0]
        # print(img_filename, depth_filename)

        img_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", config_kitti.DATA, img_filename)

        depth_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", config_kitti.DATA, depth_filename)

        source_img = Image.open(img_filename)
        depth_img = Image.open(depth_filename)
        # img width and height

        if self.transforms is not None:
            source_img = self.transforms(crop=True)(source_img)
            depth_img = self.transforms(crop=True)(depth_img)

        sparse_depth_gt_full = np.array(
            depth_img, dtype=int).astype(np.float)/256
        sparse_depth_gt_full = torch.from_numpy(sparse_depth_gt_full)
        sparse_depth_gt_full = torch.where(sparse_depth_gt_full >= config_kitti.MAX_DEPTH, torch.tensor([
                                           0], dtype=torch.float64), sparse_depth_gt_full)
        # print("sparse_depth_gt 1", sparse_depth_gt.shape)
        # print(torch.max(sparse_depth_gt), torch.min(sparse_depth_gt))
        # plt.imshow(source_img.permute(1,2,0))
        # plt.show()

        # plt.imshow(depth_img.permute(1,2,0))
        # plt.show()

        imPts, depth = self.sample_depth_img(depth_img)
        imPts_gt, depth_gt = self.sample_depth_gt_img(depth_img)

        virtual_lidar = torch.zeros(imPts.shape[0], 3)
        virtual_lidar[:, 0:2] = imPts
        virtual_lidar[:, 2] = depth

        mask = torch.zeros(source_img.shape[1:], dtype=torch.bool)
        mask[imPts[:, 0], imPts[:, 1]] = True
        # plt.imshow(mask)
        # plt.show()
        k_nn_indices = self.find_k_nearest(virtual_lidar)

        sparse_depth = torch.zeros_like(
            source_img[0, :, :].unsqueeze_(0), dtype=torch.float)

        # sparse_depth[0, imPts[:, 0], imPts[:, 1]] = torch.tensor(
        #     depth, dtype=torch.float)

        sparse_depth[0, imPts[:, 0], imPts[:, 1]
                     ] = depth.clone().detach().type(torch.float)

        # -------Generate virtual ground truth

        sparse_depth_gt = torch.zeros_like(
            source_img[0, :, :].unsqueeze_(0), dtype=torch.float)

        # sparse_depth_gt[0, imPts[:, 0], imPts[:, 1]] = torch.tensor(
        #     depth, dtype=torch.float)
        sparse_depth_gt[0, imPts[:, 0], imPts[:, 1]
                        ] = depth.clone().detach().type(torch.float)
        # sparse_depth_gt[0, imPts_gt[:, 0], imPts_gt[:, 1]] = torch.tensor(
        #     depth_gt, dtype=torch.float)

        sparse_depth_gt[0, imPts_gt[:, 0], imPts_gt[:, 1]
                        ] = depth_gt.clone().detach().type(torch.float)

        # print("sparse_depth_gt 2", sparse_depth_gt.shape)

        # plt.imshow(source_img.permute(1,2,0))
        # plt.show()

        # plt.imshow(mask)
        # plt.show()

        # plt.imshow(ann["semantic_mask"].squeeze_(0))
        # plt.show()

        # for m in ann["masks"]:
        #     plt.imshow(m)
        #     plt.show()
        # print(source_img, virtual_lidar, mask, sparse_depth, k_nn_indices, sparse_depth_gt)
        return source_img, ann, virtual_lidar, mask, sparse_depth, k_nn_indices, sparse_depth_gt, sparse_depth_gt_full, basename

    def __len__(self):
        return len(self.ids)


def get_transform(resize=False, normalize=False, crop=False):
    new_size = tuple(np.ceil(x*config_kitti.RESIZE)
                     for x in config_kitti.ORIGINAL_INPUT_SIZE_HW)
    new_size = tuple(int(x) for x in new_size)
    custom_transforms = []
    if resize:
        print("resizing samples to", new_size)
        custom_transforms.append(transforms.Resize(new_size))

    if crop:
        custom_transforms.append(
            transforms.CenterCrop(config_kitti.CROP_OUTPUT_SIZE))

    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)/255).unsqueeze(0)))
    if normalize:
        custom_transforms.append(transforms.Normalize(0.485, 0.229))
    return transforms.Compose(custom_transforms)


def get_datasets(imgs_root, depth_root, annotation, split=False, val_size=0.20, n_samples=None):
    # imgs_root, depth_root, annotation
    vkitti_dataset = vkittiDataset(
        imgs_root, depth_root, annotation, transforms=get_transform, n_samples=n_samples)
    if split:
        if val_size >= 1:
            raise AssertionError(
                "val_size must be a value within the range of (0,1)")

        len_val = math.ceil(len(vkitti_dataset)*val_size)
        len_train = len(vkitti_dataset) - len_val

        if len_train < 1 or len_val < 1:
            raise AssertionError("datasets length cannot be zero")

        indices = list(range(len(vkitti_dataset)))
        train_set = torch.utils.data.Subset(vkitti_dataset, indices[:len_train])
        val_set = torch.utils.data.Subset(vkitti_dataset, indices[len_train:])
        # train_set, val_set = torch.utils.data.random_split(
        #     vkitti_dataset, [len_train, len_val])
        return train_set, val_set
    else:
        return vkitti_dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(batch_size, imgs_root, depth_root, annotation, num_replicas, rank, split=False, val_size=0.20, n_samples=None):

    if split:
        train_set, val_set = get_datasets(
            imgs_root, depth_root, annotation, split=True, val_size=0.20, n_samples=n_samples)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=num_replicas, rank=rank, shuffle=True)

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, num_replicas=num_replicas, rank=rank, shuffle=True)

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
        dataset = get_datasets(imgs_root, depth_root, annotation,
                               split=False, val_size=0.20, n_samples=n_samples)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  collate_fn=collate_fn,
                                                  drop_last=True)
    return data_loader


#
