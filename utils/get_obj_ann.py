import json

def clear_ann_file(ann_file, output_file):

    data = json.load(open(ann_file))
    # list of images
    images = data["images"]

    annotations = data['annotations']

    # img_ids = list(map(lambda img: img["id"], images))
    img_with_ann = list(map(lambda ann: ann["image_id"], annotations))

    new_images = [im for im in images if im["id"]  in img_with_ann]

    data["images"] = new_images

    print(len(images), len(new_images))

    with open(output_file, 'w') as f:
        json.dump(data, f)


clear_ann_file("COCO_ann_loc_2.json", "COCO_ann_loc_2_obj.json")