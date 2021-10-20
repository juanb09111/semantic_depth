import os

# filename = "tmp/res/training_results_effnet_no_depthwise_30_01.txt" 
# filename = "tmp/res/training_results_1.0_data_agmentation.txt" 
# filename = "tmp/res/training_results_600.txt" 
filename = "tmp/res/training_results_480_training_120_eval.txt"

res_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))

lines = res_file.readlines()

# print(lines)

n = 2

print("batch loss")
for line in lines:

    if line.find("batch_loss:") != -1:

        res = line.strip().split(":")[-1]
        
        print(res)

print("---------------------------------------")

print("sem_seg mIoU")
for line in lines:
    if line.find("SemSeg mIoU =") != -1:

        res = line.strip().split("=")[-1]
        
        print(res)
    
print("---------------------------------------")

print("bbox map")
for line in lines:

    if line.find(" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =") != -1 and n%2 == 0:

        res = line.strip().split("=")[-1]
        
        print(res)

        n = n + 1

    elif line.find(" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =") != -1 and n%2 != 0:

        n = n + 1







