import glob
from PIL import Image
from shutil import copyfile
import os.path
import numpy as np
import statistics


# root_folder = "./results/_FuseNet_eval_results_depth_completion_1608704901.002039/" # 8000 points
root_folder = "./results/_FuseNet_eval_results_depth_completion_1610969549.02163/" # 4000 points


file_paths = glob.glob(root_folder+"/*.png")

rmse_arr = []
for image in file_paths:

    rmse = image.split("_")[-1].split(".")[0:2]
    rmse = ".".join(rmse)
    rmse_arr.append(float(rmse))

mean = statistics.mean(rmse_arr)

median = statistics.median(rmse_arr)

print("Results", "mean: {} , median: {}".format(mean,median))