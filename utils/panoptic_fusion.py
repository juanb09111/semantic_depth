import torch
import numpy as np
import os.path
import json

import matplotlib.pyplot as plt
from .apply_panoptic_mask_gpu import apply_panoptic_mask_gpu
import time


def threshold_instances(preds, threshold=0.5):

    for i in range(len(preds)):
        mask_logits, bbox_pred, class_pred, confidence = preds[i][
            "masks"], preds[i]["boxes"], preds[i]["labels"], preds[i]["scores"]

        if "ids" in preds[i].keys():
            ids = preds[i]["ids"]

        indices = (confidence > threshold).nonzero().view(1, -1)

        mask_logits = torch.index_select(mask_logits, 0, indices.squeeze(0))
        bbox_pred = torch.index_select(bbox_pred, 0, indices.squeeze(0))
        class_pred = torch.index_select(class_pred, 0, indices.squeeze(0))
        confidence = torch.index_select(confidence, 0, indices.squeeze(0))
        if "ids" in preds[i].keys():
            ids = torch.index_select(ids, 0, indices.squeeze(0))

        preds[i]["masks"] = mask_logits
        preds[i]["boxes"] = bbox_pred
        preds[i]["labels"] = class_pred
        preds[i]["scores"] = confidence
        if "ids" in  preds[i].keys():
            preds[i]["ids"] = ids

    return preds

def filter_by_class(preds, device, exclude_ins_classes=[], excluded_sem_classes=[]):
    
    for i in range(len(preds)):
        mask_logits, bbox_pred, class_pred, confidence, semantic_logits = preds[i][
            "masks"], preds[i]["boxes"], preds[i]["labels"], preds[i]["scores"], preds[i]["semantic_logits"]

        # print(semantic_logits.shape)
        if "ids" in preds[i].keys():
            ids = preds[i]["ids"]

        # filter instances by class
        included_ins_classes = torch.as_tensor([c not in exclude_ins_classes for c in class_pred])
        indices = torch.tensor(torch.where((included_ins_classes == True))[0], device=device)


        mask_logits = torch.index_select(mask_logits, 0, indices)
        bbox_pred = torch.index_select(bbox_pred, 0, indices)
        class_pred = torch.index_select(class_pred, 0, indices)
        confidence = torch.index_select(confidence, 0, indices)
        if "ids" in preds[i].keys():
            ids = torch.index_select(ids, 0, indices)

        preds[i]["masks"] = mask_logits
        preds[i]["boxes"] = bbox_pred
        preds[i]["labels"] = class_pred
        preds[i]["scores"] = confidence
        if "ids" in  preds[i].keys():
            preds[i]["ids"] = ids

        
        # filter semantic logits by class

        included_sem_classes = torch.as_tensor([c in excluded_sem_classes for c in range(semantic_logits.shape[0])])
        indices = torch.where((included_sem_classes == True))
        # print(indices)
        # preds[i]["semantic_logits"][indices] = torch.zeros_like(preds[i]["semantic_logits"][0])
        preds[i]["semantic_logits"][indices] = preds[i]["semantic_logits"][0] - 100
    return preds

def sort_by_confidence(preds):

    sorted_preds = [{"masks": torch.zeros_like(preds[i]["masks"]),
                     "boxes": torch.zeros_like(preds[i]["boxes"]),
                     "labels": torch.zeros_like(preds[i]["labels"]),
                     "scores": torch.zeros_like(preds[i]["scores"]),
                     "ids": torch.zeros_like(preds[i]["labels"]),
                     "semantic_logits": torch.zeros_like(preds[i]["semantic_logits"])} for i, _ in enumerate(preds)]

    for i in range(len(preds)):

        sorted_indices = torch.argsort(preds[i]["scores"])

        for idx, k in enumerate(sorted_indices):
            sorted_preds[i]["masks"][idx] = preds[i]["masks"][k]
            sorted_preds[i]["boxes"][idx] = preds[i]["boxes"][k]
            sorted_preds[i]["labels"][idx] = preds[i]["labels"][k]
            sorted_preds[i]["scores"][idx] = preds[i]["scores"][k]
            if "ids" in  preds[i].keys():
                sorted_preds[i]["ids"][idx] = preds[i]["ids"][k]
        
        if "ids" not in preds[i].keys():
            del sorted_preds[i]["ids"]
                

        sorted_preds[i]["semantic_logits"] = preds[i]["semantic_logits"]

    return sorted_preds


def summary(labels, thing_categories):
    count = [{"name": cat["name"], "count_obj": 0, "idx": i + 1}
             for i, cat in enumerate(thing_categories)]

    for c in count:
        indices = [i for i, x in enumerate(labels) if x == c["idx"]]
        c["count_obj"] = len(indices)

    return count




def get_MLB(preds, all_categories, thing_categories):

    all_ids = list(map(lambda cat: cat["id"], all_categories))

    batch_mlb = []

    # iterate over batch
    for _, pred in enumerate(preds):

        boxes = pred["boxes"]
        classes = pred["labels"]
        sem_logits = pred["semantic_logits"]

        non_background_objs = len(torch.nonzero(classes))
        # print("non_background_objs", non_background_objs)
        mlb_arr = torch.zeros(non_background_objs, *sem_logits[0].shape)

        i = 0
        for j, box in enumerate(boxes):

            # get current box's class
            bbox_class = classes[j]

            # Exclude background class
            if bbox_class > 0:

                # Thing categories do not include background, hence bbox_class - 1
                bbox_cat = thing_categories[bbox_class - 1]

                # Find matching id in all_ids
                cat_idx = all_ids.index(bbox_cat["id"])

                # Get the corresponding semantic tensor for the same class
                matched_sem_logits = sem_logits[cat_idx + 1]

                # Create zeroed-tensor the size of the semantic mask
                mlb = torch.zeros_like(matched_sem_logits)

                xmin, ymin, xmax, ymax = box

                xmin = xmin.to(dtype=torch.long)
                ymin = ymin.to(dtype=torch.long)
                xmax = xmax.to(dtype=torch.long)
                ymax = ymax.to(dtype=torch.long)

                mlb[ymin:ymax, xmin:xmax] = matched_sem_logits[ymin:ymax, xmin:xmax]

                mlb_arr[i] = mlb

                i += 1

        batch_mlb.append(mlb_arr)
    return batch_mlb


def get_MLA(preds):

    batch_mla = []

    for _, pred in enumerate(preds):
        classes = pred["labels"]
        masks = pred["masks"]
        boxes = pred["boxes"]

        if masks.shape[0] > 0:

            non_background_objs = len(torch.nonzero(classes))
            # print("non_background_objs", non_background_objs)
            mla_arr = torch.zeros(non_background_objs, *
                                  torch.squeeze(*masks[0]).shape)

            i = 0

            for j, mask in enumerate(masks):

                if classes[j] > 0:

                    mask_logits = torch.zeros_like(mask[0])

                    box = boxes[j]
                    xmin, ymin, xmax, ymax = box

                    xmin = xmin.to(dtype=torch.long)
                    ymin = ymin.to(dtype=torch.long)
                    xmax = xmax.to(dtype=torch.long)
                    ymax = ymax.to(dtype=torch.long)

                    box_mask_probs = mask[0][ymin:ymax, xmin:xmax]

                    box_mask_logits = torch.log(
                        box_mask_probs) - torch.log1p(-box_mask_probs)

                    mask_logits[ymin:ymax, xmin:xmax] = box_mask_logits

                    mla_arr[i] = mask_logits

                    i += 1
        else:
            mla_arr = []

        batch_mla.append(mla_arr)

    return batch_mla


def fuse_logits(MLA, MLB):

    batch_size = len(MLA)
    fl_batch = []
    for i in range(batch_size):

        mla = MLA[i]

        mlb = MLB[i]

        if len(mla) > 0 and len(mlb) > 0:

            sigmoid_mla = torch.sigmoid(mla)
            sigmoid_mlb = torch.sigmoid(mlb)

            fl = torch.mul(torch.add(sigmoid_mla, sigmoid_mlb),
                           torch.add(mla, mlb))

            fl_batch.append(fl)
        else:
            fl_batch.append(None)

    return fl_batch


def panoptic_fusion(preds, all_categories, stuff_categories, thing_categories, device, threshold_by_confidence=True, sort_confidence=True):

    print("DEVICE!!!!!!!!!!!!!!!!!!!!", device)

    inter_pred_batch = []
    sem_pred_batch = []
    summary_batch = []
    batch_size = len(preds)

    # Get list of cat in the form of (idx, supercategory)
    cat_idx = list(map(lambda cat_tuple: (
        cat_tuple[0], cat_tuple[1]["supercategory"]), enumerate(all_categories)))

    # Filter previous list with supercategory == "background"
    stuff_cat_idx_sup = list(
        filter(lambda cat_tuple: cat_tuple[1] == "background", cat_idx))

    # Get indices only for stuff categories
    stuff_cat_idx = list(
        map(lambda sutff_cat: sutff_cat[0], stuff_cat_idx_sup))

    # Add background class 0
    stuff_cat_idx = [0, *[x + 1 for x in stuff_cat_idx]]

    stuff_cat_idx = torch.LongTensor(stuff_cat_idx).to(device)

    if threshold_by_confidence:
        preds = threshold_instances(preds)

    if sort_confidence:
        preds = sort_by_confidence(preds)
    
    # detections_batch = sort_by_id(sorted_preds)
    MLA = get_MLA(preds)

    MLB = get_MLB(preds, all_categories, thing_categories)

    fused_logits_batch = fuse_logits(MLA, MLB)

    for i in range(batch_size):

        labels = preds[i]["labels"]
        
        summary_obj = summary(labels, thing_categories)
        summary_batch.append(summary_obj)

        fused_logits = fused_logits_batch[i]

        sem_logits = preds[i]["semantic_logits"]

        # TODO: check if background class 0 needs to be included in stuff_cat_idx
        stuff_sem_logits = torch.index_select(sem_logits, 0, stuff_cat_idx)

        if fused_logits is not None:

            fused_logits = fused_logits.to(device)


            # Intermediate logits
            inter_logits = torch.cat((stuff_sem_logits, fused_logits))

            # Intermediate predictions
            inter_pred = torch.argmax(inter_logits, dim=0)
            
            if "ids" in preds[i].keys():
                obj_ids = preds[i]["ids"]
                
                stuff_layers_len = len(stuff_cat_idx)
                for idx, obj_id in enumerate(obj_ids):

                    inter_pred = torch.where(inter_pred == stuff_layers_len + idx, obj_id, inter_pred)

            # Semantic predictions
            sem_pred = torch.argmax(sem_logits, dim=0)


            inter_pred_batch.append(inter_pred)

            sem_pred_batch.append(sem_pred)
        else:

            sem_pred = torch.argmax(stuff_sem_logits, dim=0)

            inter_pred_batch.append(None)

            sem_pred_batch.append(sem_pred)

    return inter_pred_batch, sem_pred_batch, summary_batch


def map_stuff(x, classes_arr, device):

    res = torch.zeros_like(x)
    default_value = torch.tensor(0).to(device)

    for c in classes_arr:
        y = torch.where(x == c, x, default_value)

        res = res + y

    return res


def map_things(x, classes_arr, device):
    res = torch.zeros_like(x)
    default_value = torch.tensor(0).to(device)

    for c in classes_arr:
        y = torch.where(x != c, x, default_value)

        res = res + y

    return res


def panoptic_canvas(inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories, device):

    batch_size = len(inter_pred_batch)

    panoptic_canvas_batch = []

    # Get list of cat in the form of (idx, supercategory)
    cat_idx = list(map(lambda cat_tuple: (
        cat_tuple[0], cat_tuple[1]["supercategory"]), enumerate(all_categories)))

    # Filter previous list with supercategory == "background"
    stuff_cat_idx_sup = list(
        filter(lambda cat_tuple: cat_tuple[1] == "background", cat_idx))

    # Get indices only for stuff categories
    stuff_cat_idx = list(
        map(lambda sutff_cat: sutff_cat[0], stuff_cat_idx_sup))

    # Add background class 0
    stuff_cat_idx = [0, *[x + 1 for x in stuff_cat_idx]]
    # print("stuff_cat_idx", stuff_cat_idx)

    # Stuff classes index in intermediate prediction, thi first len(stuff_cat_idx) correspond to stuff tensors
    stuff_in_inter_pred_idx = [x for x in range(len(stuff_cat_idx))]
    # print("stuff_in_inter_pred_idx", stuff_in_inter_pred_idx)

    default_value = torch.tensor(0).to(device)
    for i in range(batch_size):

        inter_pred = inter_pred_batch[i]
        # print(torch.max(inter_pred))
        sem_pred = sem_pred_batch[i]

        if inter_pred == None and sem_pred == None:
            panoptic_canvas_batch.append(None)

        elif inter_pred == None and sem_pred is not None:
            panoptic_canvas_batch.append(sem_pred)
        else:
            # canvases in GPU

            stuff_canvas_gpu = map_stuff(sem_pred, stuff_cat_idx, device)

            things_canvas_gpu = map_things(inter_pred, stuff_in_inter_pred_idx, device)
            panoptic_canvas_gpu = torch.where(
                things_canvas_gpu == default_value, stuff_canvas_gpu, things_canvas_gpu)
            
            panoptic_canvas_batch.append(panoptic_canvas_gpu)

    return panoptic_canvas_batch


def get_panoptic_results(images, preds, all_categories, stuff_categories, thing_categories, folder, filenames):


    batch_size = len(preds)

    inter_pred_batch, sem_pred_batch, summary_batch = panoptic_fusion(
        preds, all_categories, stuff_categories, thing_categories)

    panoptic_canvas_batch = panoptic_canvas(
        inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories)


    # TODO: panoptic_canvas_batch could be None for one of the values in the batch
    height, width = panoptic_canvas_batch[0].shape

    my_path = os.path.dirname(__file__)

    for i in range(batch_size):
        summary_arr = summary_batch[i]
        canvas = panoptic_canvas_batch[i]
        img = images[i]
        im = apply_panoptic_mask_gpu(img, canvas)
        # Move to cpu
        im = im.cpu().permute(1, 2, 0).numpy()

        file_name_basename = os.path.basename(filenames[i])
        file_name = os.path.splitext(file_name_basename)[0]

        dppi = 96
        fig, ax = plt.subplots(1, 1, figsize=(
            width/dppi, height/dppi), dpi=dppi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        c = 1
        for obj in summary_arr:
            ax.text(20, 30*c, '{}: {}'.format(obj["name"], obj["count_obj"]), style='italic',
                    bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})
            c = c + 1

        ax.imshow(im,  interpolation='nearest', aspect='auto')
        plt.axis('off')
        fig.savefig(os.path.join(
            my_path, '../{}/{}.png'.format(folder, file_name)))
        plt.close(fig)
