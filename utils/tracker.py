import torch
import config_kitti
import temp_variables
import numpy as np
import numba
from numba import jit, cuda
from .lucas_kanade import lucas_kanade_per_mask

start = config_kitti.NUM_STUFF_CLASSES + config_kitti.MAX_DETECTIONS
stop = config_kitti.MAX_DETECTIONS * (config_kitti.NUM_FRAMES + 2)
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
                    boxAArea = (new_box[2] - new_box[0] + 1) * (
                        new_box[3] - new_box[1] + 1
                    )
                    boxBArea = (prev_box[2] - prev_box[0] + 1) * (
                        prev_box[3] - prev_box[1] + 1
                    )
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    iou_matrix[i][j] = iou


def update_unmatched(ids, boxes, masks):

    # Unmatched detections
    obj = {"ids": ids, "boxes": boxes, "masks": masks}

    unmatched_obj = {"unmatched_1": [obj]}
    trk_ids_dict.update(unmatched_obj)


def init_tracker(num_ids, boxes, masks, device):

    # Init count of frames
    count = torch.tensor([1], device=device)

    # Init ids for first frame
    ids_arr = torch.tensor(
        [torch.tensor(idx + start) for idx in range(num_ids)], device=device
    )

    # Removre info from other frames if they exist
    for n in range(config_kitti.NUM_FRAMES):
        trk_ids_dict.pop("{}".format(n + 1), None)

    # Frames count starts from 1
    new_ids_obj = {
        "1": {"ids_arr": ids_arr, "unmatched": {"boxes": boxes, "masks": masks}}
    }

    # Populate active ids
    active_ids = {"active": ids_arr.cpu().data.numpy()}

    count_obj = {"count": count}

    trk_ids_dict.update(new_ids_obj)
    trk_ids_dict.update(active_ids)
    trk_ids_dict.update(count_obj)

    return ids_arr


def map_to_super_class(label, super_cat_indices):

    if label in super_cat_indices:
        return config_kitti.NUM_STUFF_CLASSES + config_kitti.NUM_THING_CLASSES + 2

    return label


def get_tracked_objects(
    prev_fname,
    new_fname,
    prev_masks,
    new_masks,
    prev_boxes,
    new_boxes,
    prev_labels,
    new_labels,
    super_cat_indices,
    iou_threshold,
    device,
):

    # print(prev_fname)
    # print(new_fname)
    new_ids = torch.zeros(
        min(config_kitti.MAX_DETECTIONS, len(new_boxes)),
        dtype=torch.long,
        device=device,
    )

    max_trk = min(config_kitti.MAX_DETECTIONS, len(new_boxes))

    if prev_boxes is None:
        return init_tracker(
            max_trk,
            new_boxes[: config_kitti.MAX_DETECTIONS],
            new_masks[: config_kitti.MAX_DETECTIONS],
            device=device,
        )

    # TODO: Handle negative samples
    if len(prev_boxes) == 0:
        return init_tracker(max_trk, [], [], device=device), None, None

    if len(new_boxes) == 0:
        return new_ids, None, None

    # Update using lk
    prev_boxes, prev_masks = lucas_kanade_per_mask(
        prev_fname, new_fname, prev_masks, prev_boxes, 0.5
    )

    # ----------------------
    # #TODO: lk on all previous frames 

    # for n in range(config_kitti.NUM_FRAMES):
    #     key = str(n+1)
    #     if key in trk_ids_dict.keys():
    #         frame_boxes = trk_ids_dict["{}".format(n + 1)]["unmatched"]["boxes"]
    #         frame_masks = trk_ids_dict["{}".format(n + 1)]["unmatched"]["masks"]
    #         if len(frame_boxes) > 0:
    #             frame_boxes, frame_masks = lucas_kanade_per_mask(
    #                 prev_fname, new_fname, frame_masks, frame_boxes, 0.5
    #             )
    #             trk_ids_dict["{}".format(n + 1)]["unmatched"]["boxes"] = frame_boxes
    #             trk_ids_dict["{}".format(n + 1)]["unmatched"]["masks"] = frame_masks
    # ------------------------------

    # Calculate iou matrix
    iou_matrix = torch.zeros(
        new_boxes[:max_trk].shape[0], prev_boxes.shape[0], device=device
    )
    blockspergrid = (
        len(prev_boxes) * len(new_boxes[:max_trk]) + (threadsperblock - 1)
    ) // threadsperblock

    mapped_prev_labels = torch.tensor(
        list(map(lambda val: map_to_super_class(val, super_cat_indices), prev_labels)),
        device=temp_variables.DEVICE,
    )
    mapped_new_labels = torch.tensor(
        list(map(lambda val: map_to_super_class(val, super_cat_indices), new_labels)),
        device=temp_variables.DEVICE,
    )

    # get_iou_matrix[blockspergrid, threadsperblock](iou_matrix, prev_boxes, new_boxes[:max_trk], prev_labels, new_labels)
    get_iou_matrix[blockspergrid, threadsperblock](
        iou_matrix,
        prev_boxes,
        new_boxes[:max_trk],
        mapped_prev_labels,
        mapped_new_labels,
    )

    # Ids not used yet

    available_ids = np.setdiff1d(ids_bank, trk_ids_dict["active"])

    # current frame
    current_frame = trk_ids_dict["count"][0] + 1

    # prev ids
    prev_ids = trk_ids_dict["{}".format(current_frame - 1)]["ids_arr"]
    # unmatched from prev_ids
    unmatched_indices = [x for x in range(max_trk)]
    for idx in range(max_trk):

        # Get matches for the current detection
        row = iou_matrix[idx, :]
        max_val = torch.max(row)
        # Get column index corresponding to prev detections
        max_val_idx = torch.argmax(row)

        if max_val > iou_threshold:
            new_ids[idx] = prev_ids[max_val_idx]
            # remove from unmatched
            unmatched_indices.remove(idx)
        else:
            new_ids[idx] = available_ids[0]
            available_ids = available_ids[1:]

    unmatched_boxes = [new_boxes[i] for i in unmatched_indices]
    unmatched_masks = [new_masks[i] for i in unmatched_indices]
    unmatched_obj = {"boxes": unmatched_boxes, "masks": unmatched_masks}
    # Update ids used in this frame, this contains all the ids used in the last config_kitti.NUM_FRAMES frames
    active_ids_arr = torch.tensor([], device=device)

    # Update frames FIFO

    # if current frame is larger than the maximum number of frames before recycling ids
    if current_frame >= config_kitti.NUM_FRAMES + 1:
        for n in range(config_kitti.NUM_FRAMES - 1):
            # update ids_arr
            trk_ids_dict["{}".format(n + 1)]["ids_arr"] = trk_ids_dict[
                "{}".format(n + 2)
            ]["ids_arr"]

            # update unmatched
            for name in ["boxes", "masks"]:
                trk_ids_dict["{}".format(n + 1)]["unmatched"][name] = trk_ids_dict[
                    "{}".format(n + 2)
                ]["unmatched"][name]

            active_ids_arr = [
                *active_ids_arr,
                *trk_ids_dict["{}".format(n + 1)]["ids_arr"],
            ]

        trk_ids_dict["{}".format(config_kitti.NUM_FRAMES)]["ids_arr"] = new_ids

        trk_ids_dict["{}".format(config_kitti.NUM_FRAMES)]["unmatched"] = unmatched_obj

        # update active ids
        active_ids_arr = [*active_ids_arr, *new_ids]

    else:
        # Increment count of frames
        trk_ids_dict["count"][0] = current_frame
        # TODO: UPDATE
        # new_ids_obj = {"{}".format(current_frame): new_ids}
        new_unmatched = []
        new_ids_obj = {
            "{}".format(current_frame): {"ids_arr": new_ids, "unmatched": unmatched_obj}
        }
        trk_ids_dict.update(new_ids_obj)
        for n in range(current_frame):
            active_ids_arr = [
                *active_ids_arr,
                *trk_ids_dict["{}".format(n + 1)]["ids_arr"],
            ]

    # Update trk_ids_dict["active"]
    active_ids_arr = torch.unique(torch.tensor(active_ids_arr))
    active_obj = {"active": active_ids_arr}
    trk_ids_dict.update(active_obj)

    return new_ids
