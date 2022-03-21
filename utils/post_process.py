import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

def post_process(ann_file, out_json):

    with open(ann_file) as f:
        ann_data = json.load(f)

    images = ann_data["images"]
    annotations = ann_data["annotations"]

    objects = {}

    for idx, im in enumerate(images):

        im_ann = annotations[idx]["segments_info"]

        for obj in im_ann:
            if obj["isthing"]:
                obj_id = obj["id"]
                score = obj["score"]
                category_id = obj["category_id"]
                cat_name = obj["cat_name"]

                if str(obj_id) not in objects.keys():
                    new_obj = {"{}".format(str(obj_id)): {
                        "id": obj_id,
                        "scores": [score],
                        "cat_ids": [category_id],
                        "cat_names": [cat_name]
                    }}
                    objects.update(new_obj)
                else:
                    objects[str(obj_id)]["scores"].append(score)
                    objects[str(obj_id)]["cat_ids"].append(category_id)
                    objects[str(obj_id)]["cat_names"].append(cat_name)

    # find max

    for obj_id in objects.keys():

        scores = objects[obj_id]["scores"]
        cat_ids = objects[obj_id]["cat_ids"]
        cat_names = objects[obj_id]["cat_names"]

        max_score = max(scores)
        max_score_idx = scores.index(max_score)

        max_score_cat_id = cat_ids[max_score_idx]
        max_score_cat_name = cat_names[max_score_idx]

        max_obj = {"max_score": max_score,
                   "max_cat_name": max_score_cat_name, "max_cat_id": max_score_cat_id}

        objects[obj_id].update(max_obj)

    # print(objects)
    # update json
    for idx, im in enumerate(images):

        im_ann = annotations[idx]["segments_info"]

        for ind, obj in enumerate(im_ann):
            if obj["isthing"]:
                obj_id = obj["id"]

                # get max
                max_score = objects[str(obj_id)]["max_score"]
                max_cat_name = objects[str(obj_id)]["max_cat_name"]
                max_cat_id = objects[str(obj_id)]["max_cat_id"]

                annotations[idx]["segments_info"][ind] = {**annotations[idx]["segments_info"][ind],
                                                          "max_score": max_score,
                                                          "max_cat_name": max_cat_name,
                                                          "max_cat_id": max_cat_id}
    
    post_process_res = {**ann_data, "annotations": annotations}
    with open(out_json, 'w') as res_file:
        json.dump(post_process_res, res_file)
    
    


# Re-draw results

def redraw_boxes(post_process_json, root_folder, out_folder):


    # print(out_folder)
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    with open(post_process_json) as f:
        ann_data = json.load(f)

    images = ann_data["images"]
    annotations = ann_data["annotations"]

    for idx, im in enumerate(images):
        im_filename = im["file_name"].split(".")[:-1]
        im_filename.append("png")
        im_filename = ".".join(im_filename)

        dst = os.path.join(out_folder, im_filename)

        im_filename = os.path.join(root_folder, im_filename)
        # img = cv2.imread(im_filename)
        img = Image.open(im_filename)
        width, height = img.size
        # height, width = img.shape[:2]

        im_ann = annotations[idx]["segments_info"]

        dppi = 96
        fig, ax = plt.subplots(1, 1, figsize=(width / dppi, height / dppi), dpi=dppi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        c = 1

        for obj in im_ann:
            if obj["isthing"]:
                obj_id = obj["id"]
                cat_name = obj["max_cat_name"]
                bbox = obj["bbox"]

                x1, y1, x2, y2 = bbox

                x_delta = x2 - x1
                y_delta = y2 - y1

                rect = patches.Rectangle(
                    (x1, y1), x_delta, y_delta, linewidth=1, edgecolor="r", facecolor="none"
                )

                # Add the patch to the Axes
                ax.add_patch(rect)
                ax.text(
                    x2,
                    y2 - 10,
                    "{}, id: {}".format(cat_name, obj_id),
                    color="white",
                    fontsize=15,
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
                )

        ax.imshow(img, interpolation="nearest", aspect="auto")
        # plt.axis('off')

        fig.savefig(dst, format="png")
        plt.close(fig)

        # print("img.shape", img.shape)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

def draw_tracks(post_process_json, root_folder, out_folder, axis_scale):

    Path(out_folder).mkdir(parents=True, exist_ok=True)

    with open(post_process_json) as f:
        ann_data = json.load(f)

    images = ann_data["images"]
    annotations = ann_data["annotations"]

    n_frames = len(images)

    im_filename =images[0]["file_name"].split(".")[:-1]
    im_filename.append("png")
    im_filename = ".".join(im_filename)


    im_filename = os.path.join(root_folder, im_filename)
    img = cv2.imread(im_filename)
    
    drawing_mask = np.zeros((n_frames*axis_scale, img.shape[1], img.shape[2])).astype(np.uint8)
    # drawing_mask = np.zeros_like(img)
    # drawing_mask = drawing_mask[:n_frames, :, :]

    color = np.random.randint(0, 255, (500, 3))
    # print(color)
    
    last_seen_dict = {}
    
    for frame_num, im in enumerate(images):
        
        drawing_mask_2 = np.zeros_like(drawing_mask)

        im_filename = im["file_name"].split(".")[:-1]
        im_filename.append("png")
        im_filename = ".".join(im_filename)

        dst = os.path.join(out_folder, im_filename)

        im_ann = annotations[frame_num]["segments_info"]
        current_obj_arr = []
        for obj in im_ann:
            if obj["isthing"]:
                
                obj_id = obj["id"]
                bbox = obj["bbox"]
                cat_name = obj["max_cat_name"]

                x1, y1, x2, y2 = bbox
                x_delta = x2 - x1

                x_center = x1 + x_delta/2

                if obj_id not in current_obj_arr:
                    current_obj_arr.append((obj_id, x_center, cat_name))
                
        
        #draw objects as points 
        current_ids = list(map(lambda curr_obj: str(curr_obj[0]), current_obj_arr))
        
        for point in current_obj_arr:
            a = frame_num*axis_scale
            b = int(np.floor(point[1]))
            obj_id = point[0]
            point_cat_name = point[2]
            
            drawing_mask = cv2.circle(drawing_mask, (b, a), 2, color[obj_id].tolist(), -1)

            cv2.putText(img=drawing_mask_2, 
                    text="{} - {}".format(point_cat_name, obj_id), 
                    org=(b, a+10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=0.5, 
                    color=color[obj_id].tolist(),
                    thickness=1)

            if frame_num > 0:
                if str(obj_id) in last_seen_dict.keys():
                    d, c, _ = last_seen_dict["{}".format(obj_id)]
                    drawing_mask = cv2.line(drawing_mask, (b, a),(c,d), color[obj_id].tolist(), 2)
            
            obj_last_seen = {"{}".format(obj_id): (frame_num*axis_scale, b, point_cat_name)}
            last_seen_dict.update(obj_last_seen)
        
        for last_seen in last_seen_dict.keys():
            if last_seen not in current_ids:
                d, c, cat_name = last_seen_dict["{}".format(last_seen)]
                cv2.putText(img=drawing_mask_2, 
                    text="{} - {}".format(cat_name, last_seen), 
                    org=(c, d+10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=0.5, 
                    color=color[int(last_seen)].tolist(),
                    thickness=1)

                


            
        img = cv2.add(drawing_mask, drawing_mask_2)
        # print(drawing_mask.shape)
        # cv2.imshow('image', drawing_mask)
        # cv2.waitKey(0)
        cv2.imwrite(dst, img)



filename = "results/ObjTrck_improve/inference_PanopticSeg_supercat_reverse_5frames_memory_no_recycle_show_unmatched_no_box/pred.json"
out_filename = "results/ObjTrck_improve/inference_PanopticSeg_supercat_reverse_5frames_memory_no_recycle_show_unmatched_no_box/pred_post_process.json"

res_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", out_filename)

post_process(res_file, out_json)



root_folder = "results/ObjTrck_improve/inference_PanopticSeg_supercat_reverse_5frames_memory_no_recycle_show_unmatched_no_box_vis"

root_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", root_folder)

out_folder = "results/ObjTrck_improve/inference_PanopticSeg_supercat_reverse_5frames_memory_no_recycle_show_unmatched_no_box_post_process_vis"

out_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", out_folder)


redraw_boxes(out_json, root_folder, out_folder)

# print(out_json)
axis_scale = 10
out_folder = os.path.join(out_folder, "tracks_{}".format(axis_scale))

draw_tracks(out_json, root_folder, out_folder, axis_scale)