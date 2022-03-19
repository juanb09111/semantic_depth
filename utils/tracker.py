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


def init_tracker(num_ids, boxes, masks, labels, device):

    # Init count of frames
    count = torch.tensor([1], device=device)

    # Init ids for first frame
    ids_arr = torch.tensor(
        [torch.tensor(idx + start) for idx in range(num_ids)], device=device
    )
    print("seed ids:", ids_arr)
    # Removre info from other frames if they exist
    for n in range(config_kitti.NUM_FRAMES):
        trk_ids_dict.pop("{}".format(n + 1), None)

    # Frames count starts from 1
    new_ids_obj = {
        "1": {"ids_arr": ids_arr, "unmatched": {"boxes": boxes, "masks": masks, "labels": labels, "ids": ids_arr}}
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
            new_labels[: config_kitti.MAX_DETECTIONS],
            device=device,
        )

    # TODO: Handle negative samples
    if len(prev_boxes) == 0:
        return init_tracker(max_trk, [], [], [], device=device)

    if len(new_boxes) == 0:
        return new_ids

    # Update/init variables -----------------------

    # unmatched
    # unmatched indices from latest frame
    unmatched_indices = [x for x in range(max_trk)]
    matched_indices = []

    # current frame
    current_frame = trk_ids_dict["count"][0] + 1

    # -------------------------------------------
    frame_limit = min(config_kitti.NUM_FRAMES, current_frame - 1)
    print("frame limit is: ", frame_limit)
    # TODO: reverse Loop over prev frames and update with lk -----------------------------------------------------

    for n in range(frame_limit, 0, -1):
        
        print("finding matches from frame {}.......".format(n))
        # lukas kanade, move all of previous unmatched detections to the latest frame
        trk_ids_dict["{}".format(n)]["unmatched"]["boxes"], trk_ids_dict["{}".format(n)]["unmatched"]["masks"] = lucas_kanade_per_mask(
            prev_fname, new_fname, trk_ids_dict["{}".format(
                n)]["unmatched"]["masks"],
            trk_ids_dict["{}".format(n)]["unmatched"]["boxes"],
            0.5
        )
        # end lukas kanade
        # TODO: Check that order is kept after lk.
        unmatched = trk_ids_dict["{}".format(n)]["unmatched"]
        unmatched_boxes = unmatched["boxes"]
        # unmatched_masks = unmatched["masks"]
        unmatched_labels = unmatched["labels"]
        unmatched_ids = unmatched["ids"]  # unmatched ids from previous frames

        print("unmatched ids from frame {} (before): {}".format(n, unmatched_ids))

        # calculate only if there are unmatched items
        if len(unmatched_ids) > 0 and len(unmatched_indices) > 0:
            # Calculate iou matrix
            iou_matrix = torch.zeros(
                new_boxes[:max_trk].shape[0], unmatched_boxes.shape[0], device=device
            )
            blockspergrid = (
                len(unmatched_boxes) *
                len(new_boxes[:max_trk]) + (threadsperblock - 1)
            ) // threadsperblock

            mapped_unmatched_labels = torch.tensor(
                list(map(lambda val: map_to_super_class(
                    val, super_cat_indices), unmatched_labels)),
                device=temp_variables.DEVICE,
            )
            mapped_new_labels = torch.tensor(
                list(map(lambda val: map_to_super_class(
                    val, super_cat_indices), new_labels)),
                device=temp_variables.DEVICE,
            )

            # get_iou_matrix[blockspergrid, threadsperblock](iou_matrix, prev_boxes, new_boxes[:max_trk], prev_labels, new_labels)
            get_iou_matrix[blockspergrid, threadsperblock](
                iou_matrix,
                unmatched_boxes,
                new_boxes[:max_trk],
                mapped_unmatched_labels,
                mapped_new_labels
            )

            # update iou matrix, set already matched new_detections to 0
            for matched_ind in matched_indices:
                iou_matrix[matched_ind, :] = 0

            # Ids not used yet
            available_ids = np.setdiff1d(ids_bank, trk_ids_dict["active"])

            # TODO: Change to unmatched prev ids
            # DONE
            matched_prev_indices = []
            unamtched_prev_indices = [x for x in range(iou_matrix.shape[1])]
            print("unamtched_prev_indices", unamtched_prev_indices)
            for idx in range(max_trk):

                # Get matches for the current detection
                row = iou_matrix[idx, :]
                max_val = torch.max(row)
                # Get column index corresponding to prev detections
                max_val_idx = torch.argmax(row)

                if max_val > iou_threshold and idx not in matched_indices:
                    new_ids[idx] = unmatched_ids[max_val_idx]
                    matched_prev_indices.append(max_val_idx)
                    unamtched_prev_indices.remove(max_val_idx)
                    # update iou to avoid double hit
                    iou_matrix[:, max_val_idx] = 0

                    # remove indices from new detections from unmatched
                    unmatched_indices.remove(idx)
                    matched_indices.append(idx)


                elif idx not in matched_indices:  # as long as the new det has not already been matched --> assign new id
                    new_ids[idx] = available_ids[0]
                    available_ids = available_ids[1:]

            # Update prev unmatch and
            # for unmatched_prev_idx in unamtched_prev_indices:
            for key in ["boxes", "masks", "labels", "ids"]:
                arr = trk_ids_dict["{}".format(n)]["unmatched"][key]
                indices = torch.tensor(unamtched_prev_indices, device=device, dtype=torch.long)
                print("indices to keep (unmatched from prev frame): {}".format(indices))
                trk_ids_dict["{}".format(n)]["unmatched"][key] = torch.index_select(arr, 0, indices)
                

            print("unmatched ids from frame {} - after: {}".format(n, trk_ids_dict["{}".format(n)]["unmatched"]["ids"]))

    unmatched_obj = {"boxes": new_boxes[: config_kitti.MAX_DETECTIONS],
                     "masks": new_masks[: config_kitti.MAX_DETECTIONS],
                     "labels": new_labels[: config_kitti.MAX_DETECTIONS],
                     "ids": new_ids}
    print("new frame ids:", new_ids)
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
            for name in ["boxes", "masks", "labels", "ids"]:
                trk_ids_dict["{}".format(n + 1)]["unmatched"][name] = trk_ids_dict[
                    "{}".format(n + 2)
                ]["unmatched"][name]

            active_ids_arr = [
                *active_ids_arr,
                *trk_ids_dict["{}".format(n + 1)]["ids_arr"],
            ]

        trk_ids_dict["{}".format(config_kitti.NUM_FRAMES)]["ids_arr"] = new_ids

        trk_ids_dict["{}".format(config_kitti.NUM_FRAMES)
                     ]["unmatched"] = unmatched_obj

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
