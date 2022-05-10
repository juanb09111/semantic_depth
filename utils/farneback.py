import cv2
import numpy as np
import os
import torch
import temp_variables

import matplotlib.pyplot as plt



import time

def flow_det(prev_masks, prev_boxes, flow, confidence=0.5):

    pred_masks = torch.zeros_like(prev_masks)
    pred_boxes = torch.zeros_like(prev_boxes)

    for idx, mask in enumerate(prev_masks):

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

        drawing_mask = np.zeros_like(bin_mask_np)
        
        mask_coords = np.where(bin_mask_np > 0)
        y, x = mask_coords
        # print(len(mask_coords[0]), flow.shape, mask_coords)
        # print(flow[mask_coords[0][0], mask_coords[1][0], :])
        fx, fy = flow[y, x, :].T

        new_coords = np.vstack([y + fy, x + fx]).T.reshape(-1, 2)
        new_coords = np.int32(new_coords + 0.5)

        min_x = np.min(new_coords[:, 1]) 
        min_y = np.min(new_coords[:, 0])
        max_x = np.max(new_coords[:, 1])
        max_y = np.max(new_coords[:, 0])

        mask_coords = np.asarray(mask_coords).T.reshape(-1,2)

        # delta_x = new_coords[:, 1] - mask_coords[:, 1]
        # delta_y = new_coords[:, 0] - mask_coords[:, 0]

        # delta_x_mean = np.mean(delta_x)
        # delta_y_mean = np.mean(delta_y)

        # _x1 = int(prev_boxes[idx][0] + delta_x_mean)
        # _y1 = int(prev_boxes[idx][1] + delta_y_mean)

        # _x2 = int(prev_boxes[idx][2] + delta_x_mean)
        # _y2 = int(prev_boxes[idx][3] + delta_y_mean)

        # update pred boxes
        pred_boxes[idx][0] = min_x
        pred_boxes[idx][1] = min_y
        pred_boxes[idx][2] = max_x
        pred_boxes[idx][3] = max_y

        coord = list(zip(mask_coords, new_coords))
        h, w = drawing_mask.shape
        
        for mask_coord, new_coord in coord:
            # print(mask_coord, new_coord)
            y1, x1 = mask_coord
            y2, x2 = new_coord
            
            if y2 < h and x2 < w:
                drawing_mask[y2, x2] = bin_mask_np[y1, x1]

        #update masks
        pred_masks[idx] = torch.tensor(drawing_mask)

        # #visualization
        # drawing_mask= cv2.cvtColor(drawing_mask, cv2.COLOR_GRAY2BGR)

        # cv2.rectangle(drawing_mask,(min_x,min_y), (max_x,max_y),(0,255,0), 2)

        # # Get prev box coordinates
        # prev_x1 = int(np.floor(prev_boxes[idx][0].cpu()))
        # prev_y1 = int(np.floor(prev_boxes[idx][1].cpu()))
        # prev_x2 = int(np.floor(prev_boxes[idx][2].cpu()))
        # prev_y2 = int(np.floor(prev_boxes[idx][3].cpu()))

        # cv2.rectangle(drawing_mask,(prev_x1, prev_y1), (prev_x2, prev_y2),(0,0,255), 2)

        # cv2.imshow('image', drawing_mask)
        # cv2.waitKey(0)

        # print(new_coords.shape)
        # print("fx", fx.shape, fy.shape)
        # print(len(fx), len(fy))
    return pred_boxes, pred_masks

def draw_flow(img, flow, step=16, to_gray=False):
    h, w = img.shape[:2]
    #TODO: Get grid within mask
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    #TODO Get flow within mask
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    #TODO: Calculate avg within mask, move mask
    # create zeros mask
    # assign x, y to x+dx, y+dy
    if to_gray:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def dense_optical_flow(method, 
    dst, 
    prev_img_fname, 
    next_image_fname,
    prev_masks,
    prev_boxes,
    confidence, 
    dict_params={}, 
    params=[], 
    to_gray=False):
    # Read the video and first frame

    prev_img = cv2.imread(prev_img_fname)
    next_img = cv2.imread(next_image_fname)
    frame_copy = next_img

   

    # crate HSV & make Value a constant
    hsv = np.zeros_like(prev_img)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = method(prev_img, next_img, None, *params, **dict_params)

    pred_boxes, pred_masks = flow_det(prev_masks, prev_boxes, flow, confidence=confidence)

    # flow_im = draw_flow(next_img, flow, to_gray=to_gray)
    # fname = next_image_fname.split("/")[-1].split(".")[0]
    # cv2.imwrite("results/{}/{}.png".format(dst, fname), flow_im)
    # # Encoding: convert the algorithm's output into Polar coordinates
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # # Use Hue and Value to encode the Optical Flow
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # # Convert HSV image into BGR for demo
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow("frame", frame_copy)
    # cv2.waitKey(0)
    # cv2.imshow("optical flow", bgr)
    # cv2.waitKey(0)
    return pred_boxes, pred_masks

def cal_flow(algorithm, 
    prev_img_fname, 
    next_image_fname,
    prev_masks,
    prev_boxes,
    confidence=0.5):

    if algorithm == 'lucaskanade_dense':
        method = cv2.optflow.calcOpticalFlowSparseToDense
        dst = "lucaskanade_dense"
        pred_boxes, pred_masks = dense_optical_flow(method, 
            dst, 
            prev_img_fname, 
            next_image_fname,
            prev_masks,
            prev_boxes,
            confidence, 
            {}, 
            params=[], 
            to_gray=True)
    elif algorithm == 'farneback':
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters
        dst = "farneback"
        pred_boxes, pred_masks = dense_optical_flow(method, 
            dst, 
            prev_img_fname, 
            next_image_fname,
            prev_masks,
            prev_boxes,
            confidence, 
            {}, 
            params, 
            to_gray=True)
    elif algorithm == "rlof":
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        dst = "rlof"
        rlof_params = dict(interp_type= cv2.optflow.INTERP_GEO)
        pred_boxes, pred_masks = dense_optical_flow(method, 
            dst, 
            prev_img_fname, 
            next_image_fname,
            prev_masks,
            prev_boxes,
            confidence, 
            rlof_params, 
            params=[], 
            to_gray=False)
    
    return pred_boxes, pred_masks

