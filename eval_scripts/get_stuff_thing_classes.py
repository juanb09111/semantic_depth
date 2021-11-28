import json

def get_stuff_thing_classes(ann_file):

    with open(ann_file) as hasty_file:
        # read file
        data = json.load(hasty_file)
        all_categories = data["categories"]

    stuff_categories = list(
        filter(lambda cat: cat["supercategory"] != "object", all_categories))

    thing_categories = list(
        filter(lambda cat: cat["supercategory"] == "object", all_categories))

    return all_categories, stuff_categories, thing_categories