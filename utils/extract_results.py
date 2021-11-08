import os
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir="runs/Panoptic_crop_120s_200_1300")


# filename = "tmp/res/training_results_effnet_no_depthwise_30_01.txt" 
# filename = "tmp/res/training_results_1.0_data_agmentation.txt" 
# filename = "tmp/res/training_results_600.txt" 
filename = "tmp/res/eval_results.txt"

res_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename))

lines = res_file.readlines()

# print(lines)

n = 2

# print("batch loss")
# for line in lines:

#     if line.find("batch_loss:") != -1:

#         res = line.strip().split(":")[-1]
        
#         print(res)

# print("---------------------------------------")

# print("sem_seg mIoU")
# for line in lines:
#     if line.find("SemSeg mIoU =") != -1:

#         res = line.strip().split("=")[-1]
        
#         print(res)
    
# print("---------------------------------------")

res_bbox = []
res_segm = []

for line in lines:

    if line.find(" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =") != -1 and n%2 == 0:

        res = line.strip().split("=")[-1]
        res_bbox.append(res)


        n = n + 1

    elif line.find(" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =") != -1 and n%2 != 0:
        res = line.strip().split("=")[-1]
        res_segm.append(res)
        n = n + 1


for idx, n in enumerate(res_bbox):
    # # print(res_bbox, res_segm)
    # writer.add_scalars('coco_eval', {'bbox':float(res_bbox[idx]),
    #                                 'segm':float(res_segm[idx])}, idx)
    writer.add_scalar("coco_eval/train/bbox", float(res_bbox[idx]), idx)
    writer.add_scalar("coco_eval/train/sgm", float(res_segm[idx]), idx)

writer.close()


