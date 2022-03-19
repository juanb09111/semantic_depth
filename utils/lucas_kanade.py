import cv2
import numpy as np
import os
import torch


import matplotlib.pyplot as plt

import temp_variables

import time

import config_kitti



def lucas_kanade_per_mask(
    prev_img_fname,
    next_image_fname,
    prev_masks,
    prev_boxes,
    confidence,
    find_keypoints=False,
    save_as="res",
):

    

    pred_boxes = torch.zeros_like(prev_boxes)
    pred_masks = torch.zeros_like(prev_masks)
    prev_img = cv2.imread(prev_img_fname)
    next_img = cv2.imread(next_image_fname)

    # Create a mask image for drawing purposes
    drawing_mask = np.zeros_like(prev_img)

    for idx, mask in enumerate(prev_masks):

        # create single mask

        bin_mask = torch.where(
            mask > confidence,
            torch.tensor([1], device=temp_variables.DEVICE),
            torch.tensor([0], device=temp_variables.DEVICE),
        )
        bin_mask = bin_mask.squeeze_()
        bin_mask = torch.where(
            bin_mask > 0,
            torch.tensor([255], device=temp_variables.DEVICE),
            torch.tensor([0], device=temp_variables.DEVICE),
        )

        bin_mask_np = bin_mask.cpu().numpy().astype(np.uint8)
        # plt.figure(0)
        # plt.imshow(bin_mask_np)
        # plt.savefig("lucas_kanade/masks_t/mask{}.png".format(idx))

        # lucas kanade
        lk_params = dict(
            winSize=(100, 100),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
        )

        feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        color = np.random.randint(0, 255, (100, 3))

        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_img_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        # roi_mask = np.zeros_like(prev_img_gray)
        # print(bin_mask_np.shape, roi_mask.shape, type(bin_mask_np[100][100]), type(roi_mask[100][100]))

        # roi_mask[:, :] = 255
        p0 = None

        if find_keypoints:
            # print("here!!!!!!")
            p0 = cv2.goodFeaturesToTrack(
                prev_img_gray, mask=bin_mask_np, **feature_params
            )

        if p0 is None or len(p0) == 0:
            coords = np.where(bin_mask_np > 0)

            x_coords = coords[1]
            y_coords = coords[0]

            num_coords = len(x_coords)
            # print("num_coords",num_coords)
            max_points = np.min([int(np.floor(num_coords / 2)), 50])

            rand_inds = np.random.randint(
                low=0, high=num_coords - 1, size=int(max_points)
            )

            new_coords = list(
                map(lambda ind: [[x_coords[ind], y_coords[ind]]], rand_inds)
            )
            new_coords = np.asarray([new_coords]).astype(np.float32)
            new_coords = np.reshape(new_coords, (-1, 1, 2))

            p0 = new_coords
        # print("-------------------------------------------", new_coords.shape)
        # for i in range(p0.shape[0]):
        #     p0_ = p0[:i, :, :]
        #     start = time.time()
        #     p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img_gray, next_img_gray, p0_, None, **lk_params)
        #     end = time.time()
        #     print(end - start)
        # print("----------------------------------------------")
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_img_gray, next_img_gray, p0, None, **lk_params
        )

        # th = 1
        # print("len th", (st==1), len(p1), len(p0), st)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # while len(st==th)==0:
        #     th = th - 0.1
        #     good_new = p1[st==th]
        #     good_old = p0[st==th]
        # print("th", th)

        # Find deltas
        # print("good", len(good_new), len(good_old))
        new_x = list(map(lambda coord: coord[0], good_new))
        old_x = list(map(lambda coord: coord[0], good_old))

        delta_x = np.array(new_x) - np.array(old_x)
        # print(delta_x, new_x, old_x)
        new_y = list(map(lambda coord: coord[1], good_new))
        old_y = list(map(lambda coord: coord[1], good_old))

        delta_y = np.array(new_y) - np.array(old_y)

        delta_x_mean = np.mean(delta_x)
        delta_y_mean = np.mean(delta_y)

        # lk Predicted box
        # print("debug", prev_boxes[idx][0], delta_x_mean)

        _x1 = int(prev_boxes[idx][0] + delta_x_mean)
        _y1 = int(prev_boxes[idx][1] + delta_y_mean)

        _x2 = int(prev_boxes[idx][2] + delta_x_mean)
        _y2 = int(prev_boxes[idx][3] + delta_y_mean)

        # update pred boxes
        pred_boxes[idx][0] = _x1
        pred_boxes[idx][1] = _y1
        pred_boxes[idx][2] = _x2
        pred_boxes[idx][3] = _y2

        # update mask
        # translate coords
        translated_coords = [[], []]

        coords = np.where(bin_mask_np > 0)
        translated_coords[0] = (coords[0] + delta_y_mean).astype(np.int16)
        translated_coords[1] = (coords[1] + delta_x_mean).astype(np.int16)

        new_mask = np.zeros_like(bin_mask_np, dtype=np.uint8)
        translated_coords[0] = np.where(
            translated_coords[0] >= new_mask.shape[0],
            new_mask.shape[0] - 1,
            translated_coords[0],
        )
        translated_coords[1] = np.where(
            translated_coords[1] >= new_mask.shape[1],
            new_mask.shape[1] - 1,
            translated_coords[1],
        )

        # coords = np.where(bin_mask_np > 0)
        new_mask[tuple(translated_coords)] = 1
        # plt.figure(1)
        # plt.imshow(new_mask)
        # plt.savefig("lucas_kanade/masks_t/mask{}_1.png".format(idx))
        pred_masks[idx] = torch.tensor(new_mask)

    #     for i,(new,old) in enumerate(zip(good_new,good_old)):
    #         a,b = list([int(x) for x in np.floor(new.ravel())])
    #         c,d = list([int(x) for x in np.floor(old.ravel())])
    #         drawing_mask = cv2.line(drawing_mask, (a,b),(c,d), color[i].tolist(), 2)
    #         next_img = cv2.circle(next_img,(a,b),5,color[i].tolist(),-1)

    #     cv2.rectangle(drawing_mask,(_x1,_y1), (_x2,_y2),(0,255,0), 2)

    #     # Get prev box coordinates
    #     prev_x1 = int(np.floor(prev_boxes[idx][0].cpu()))
    #     prev_y1 = int(np.floor(prev_boxes[idx][1].cpu()))
    #     prev_x2 = int(np.floor(prev_boxes[idx][2].cpu()))
    #     prev_y2 = int(np.floor(prev_boxes[idx][3].cpu()))

    #     cv2.rectangle(drawing_mask,(prev_x1, prev_y1), (prev_x2, prev_y2),(0,0,255), 2)

    #     img = cv2.add(next_img, drawing_mask)
    # # print(save_as)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.imwrite("lucas_kanade/progress_unmatched/{}.png".format(save_as), img)
    # print(pred_boxes[0])
    return pred_boxes, pred_masks
