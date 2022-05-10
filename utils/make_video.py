# %%
import re
import os.path
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import glob
import config_kitti

data_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../{}'.format(config_kitti.VIDEO_CONTAINER_FOLDER))

images = []


for infile in sorted(glob.glob('{}/*.png'.format(data_dir))):
    images.append(os.path.basename(infile))

# images.reverse()

frame = cv2.imread(os.path.join(data_dir, images[0]))
height, width, layers = frame.shape

out = cv2.VideoWriter(config_kitti.VIDEO_OUTOUT_FILENAME, cv2.VideoWriter_fourcc(
    *'DIVX'), config_kitti.FPS, (width, height))


for img in images:
    out.write(cv2.imread(os.path.join(data_dir, img)))
cv2.destroyAllWindows()
out.release()
# %%
