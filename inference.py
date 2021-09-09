import config
import temp_variables
import constants
import models

from utils.tensorize_batch import tensorize_batch

import os
import os.path
from pathlib import Path
import torch

from datetime import datetime


from utils import  panoptic_fusion, get_datasets
from utils.show_segmentation import apply_semantic_mask_gpu, apply_instance_masks, save_fig



device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

temp_variables.DEVICE = device
torch.cuda.empty_cache()

all_categories, stuff_categories, thing_categories = panoptic_fusion.get_stuff_thing_classes()


def view_masks(model,
               data_loader_val,
               num_classes,
               weights_file,
               result_type,
               folder,
               confidence=0.5):

    # Create folde if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    # load weights
    model.load_state_dict(torch.load(weights_file))
    # move model to the right device
    model.to(device)
    for images, anns in data_loader_val:
        images = list(img for img in images)
        images = tensorize_batch(images, device)
        file_names = list(map(lambda ann: ann["file_name"], anns))
        model.eval()
        with torch.no_grad():

            outputs = model(images)

            if result_type == "panoptic":
                panoptic_fusion.get_panoptic_results(
                    images, outputs, all_categories, stuff_categories, thing_categories, folder, file_names)
                torch.cuda.empty_cache()
            else: 
                for idx, output in enumerate(outputs):
                    file_name = file_names[idx]
                    if result_type == "instance":
                        im = apply_instance_masks(images[idx], output['masks'], 0.5)

                    elif result_type == "semantic":
                        logits = output["semantic_logits"]
                        mask = torch.argmax(logits, dim=0)
                        im = apply_semantic_mask_gpu(images[idx], mask, config.NUM_STUFF_CLASSES + config.NUM_THING_CLASSES)
                    
                    save_fig(im, folder, file_name)

                    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    test_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), config.TEST_DIR)
    data_loader_test = get_datasets.get_dataloaders(
        config.BATCH_SIZE, test_dir, is_test_set=True)

    confidence = 0.5
    model = models.get_model()

    if config.INSTANCE:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "instance",
                   '{}{}_{}_results_instance_{}'.format(constants.INFERENCE_RESULTS,
                       config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)

    if config.SEMANTIC:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "semantic",
                   '{}{}_{}_results_semantic_{}'.format(constants.INFERENCE_RESULTS,
                       config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)

    if config.PANOPTIC:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "panoptic",
                   '{}{}_{}_results_panoptic_{}'.format(constants.INFERENCE_RESULTS,
                       config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)
