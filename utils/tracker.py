import torch
import config_kitti
import temp_variables
import numpy as np
import numba
from numba import jit, cuda 

start = config_kitti.NUM_STUFF_CLASSES + config_kitti.MAX_DETECTIONS
stop = config_kitti.MAX_DETECTIONS*(config_kitti.NUM_FRAMES+2)
ids_bank = torch.tensor(list(range(start, stop)))
trk_ids_dict = {}

threadsperblock = 128



@cuda.jit
def get_iou_matrix(iou_matrix, prev_boxes, new_boxes, prev_labels, new_labels):

    for i, new_box in enumerate(new_boxes):
        for j, prev_box in enumerate(prev_boxes):
            if new_labels[i] != prev_labels[j]:
                # ciou_time = 0
                iou_matrix[i][j] = 0

            else:
                # min x of intersection
                xA = max(prev_box[0], new_box[0])
                # min y of intersection
                yA = max(prev_box[1], new_box[1])

                # max x of intersection
                xB = min(prev_box[2], new_box[2])
                # max y of intersection
                yB = min(prev_box[3], new_box[3])
                # compute the area of intersection rectangle
                if xA > xB or yA > yB:
                    # ciou_time = 0
                    iou_matrix[i][j] = 0
                else:
                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                    boxAArea = (new_box[2] - new_box[0] + 1) * (new_box[3] - new_box[1] + 1)
                    boxBArea = (prev_box[2] - prev_box[0] + 1) * (prev_box[3] - prev_box[1] + 1)
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    iou_matrix[i][j] = iou



def init_tracker(num_ids, device):

    # Init count of frames
    count = torch.tensor([1], device=device)

    # Init ids for first frame
    ids_arr = torch.tensor([torch.tensor(idx + start) for idx in range(num_ids)], device=device)
    
    # Removre info from other frames if they exist
    for n in range(config_kitti.NUM_FRAMES):
        trk_ids_dict.pop("{}".format(n+1), None)

    # Frames count start from 1
    new_ids_obj = {"1": ids_arr}

    # Populate active ids 
    active_ids = {"active": ids_arr.cpu().data.numpy()}

    count_obj = {"count": count}

    trk_ids_dict.update(new_ids_obj)
    trk_ids_dict.update(active_ids)
    trk_ids_dict.update(count_obj)

    return ids_arr


def get_tracked_objects(prev_boxes, new_boxes, prev_labels, new_labels, iou_threshold, device):


    new_ids= torch.zeros(min(config_kitti.MAX_DETECTIONS, len(new_boxes)), dtype=torch.long, device=device)

    max_trk = min(config_kitti.MAX_DETECTIONS, len(new_boxes))

    if prev_boxes is None:
        return init_tracker(max_trk, device=device) 
    
    
    if len(prev_boxes) == 0:
        return init_tracker(max_trk, device=device)

    if len(new_boxes) == 0:
        return new_ids


    # Calculate iou matrix
    iou_matrix = torch.zeros(new_boxes[:max_trk].shape[0], prev_boxes.shape[0], device=device)
    blockspergrid = (len(prev_boxes)*len(new_boxes[:max_trk]) + (threadsperblock - 1)) // threadsperblock
    get_iou_matrix[blockspergrid, threadsperblock](iou_matrix, prev_boxes, new_boxes[:max_trk], prev_labels, new_labels)


    # Ids not used yet

    available_ids = np.setdiff1d(ids_bank, trk_ids_dict["active"])

    # current frame
    current_frame = trk_ids_dict["count"][0] + 1

    # prev ids 
    prev_ids = trk_ids_dict["{}".format(current_frame - 1)]

    for idx in range(max_trk):

        # Get matches for the current detection
        row = iou_matrix[idx, :]
        max_val = torch.max(row)
        max_val_idx = torch.argmax(row)

        if max_val > iou_threshold:
            new_ids[idx] = prev_ids[max_val_idx]
        else:
            new_ids[idx] = available_ids[0]
            available_ids = available_ids[1:]

    # Update ids used in this frame, this contains all the ids used in the last config_kitti.NUM_FRAMES frames
    active_ids_arr = torch.tensor([], device=device)

    # Update frames FIFO

    # if current frame is larger than the maximum number of frames before recycling ids
    if current_frame >= config_kitti.NUM_FRAMES + 1:
        for n in range(config_kitti.NUM_FRAMES - 1):
            trk_ids_dict["{}".format(n+1)] = trk_ids_dict["{}".format(n+2)]
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]

        trk_ids_dict["{}".format(config_kitti.NUM_FRAMES)] = new_ids

        # update active ids
        active_ids_arr = [*active_ids_arr, *new_ids]
    
    else:
        # Increment count of frames
        trk_ids_dict["count"][0] = current_frame
        new_ids_obj = {"{}".format(current_frame): new_ids}
        trk_ids_dict.update(new_ids_obj)
        for n in range(current_frame):
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]

    # Update trk_ids_dict["active"]
    active_ids_arr = torch.unique(torch.tensor(active_ids_arr))
    active_obj = {"active": active_ids_arr}
    trk_ids_dict.update(active_obj)


    return new_ids
