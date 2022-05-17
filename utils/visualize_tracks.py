
# %%
import enum
import re
import os.path
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import glob
import config_kitti
from matplotlib import pyplot as plt
from pathlib import Path



panoptic = "results/ObjTrck_improve/inference_PanopticSeg_supercat_reverse_5frames_memory_no_recycle_show_unmatched_no_box_post_process_vis"
tracks = "results/ObjTrck_improve/inference_PanopticSeg_supercat_reverse_5frames_memory_no_recycle_show_unmatched_no_box_post_process_vis/tracks_10"

out = os.path.join(panoptic, "overlay_tracks")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../{}'.format(out))
Path(out).mkdir(parents=True, exist_ok=True)

data_dir_panoptic = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../{}'.format(panoptic))

data_dir_tracks = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../{}'.format(tracks))

images_panoptic = []
images_tracks = []

for infile in sorted(glob.glob('{}/*.png'.format(data_dir_panoptic))):
    images_panoptic.append(os.path.basename(infile))

for infile in sorted(glob.glob('{}/*.png'.format(data_dir_tracks))):
    images_tracks.append(os.path.basename(infile))

images_panoptic.reverse()
images_tracks.reverse()

for idx, (im_panoptic, im_track) in enumerate(list(zip(images_panoptic, images_tracks))):
    
    img_panoptic = cv2.imread(os.path.join(data_dir_panoptic, im_panoptic))
    img_track = cv2.imread(os.path.join(data_dir_tracks, im_track))
    track_heght, _, _ = img_track.shape
    
    drawing_mask = np.zeros_like(img_panoptic)

    drawing_mask[:track_heght, :, :] = img_track[:track_heght, :, :]

    drawing_mask = cv2.addWeighted(img_panoptic, 0.4, drawing_mask, 1, 10)
    # print(np.where(img_track == 0, img_panoptic, img_track))
    # # plt.imshow(img_track, interpolation='nearest')
    # # plt.show()

    # cv2.imshow('image', drawing_mask)
    # cv2.waitKey(0)
    # print(out)
    dst = os.path.join(out, im_panoptic)
    cv2.imwrite(dst, drawing_mask)
    



