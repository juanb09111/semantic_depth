#!/bin/bash
#
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
#SBATCH -J JP_JOB
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=res/res_semseg_depth_%a.txt
#SBATCH --error=res/err_semseg_depth_%a.txt
#
#!/bin/bash
#SBATCH --job-name=lagosben
#SBATCH --account=project_2003593
#SBATCH --partition=gpu
#SBATCH --time=2-23:59:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:4
#SBATCH  --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-user=juanpablo.lagosbenitez@tuni.fi
#
# These commands will be executed on the compute node:

# Load all modules you need below. Edit these to your needs.

#module load matlab
#module lodad python
module load CUDA/9.0
# conda activate pynoptorch
source activate pynoptorch



# if some error happens in the initialation of parallel process then you can
# get the debug info. This can easily increase the size of out.txt.
# export NCCL_DEBUG=INFO  # comment it if you are not debugging distributed parallel setup

# export NCCL_DEBUG_SUBSYS=ALL # comment it if you are not debugging distributed parallel setup

# find the ip-address of one of the node. Treat it as master
ip1=`hostname -I | awk '{print $2}'`
echo $ip1

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

#inference params
MODEL_NAME=$1
BATCH_SIZE=$2
CHECKPOINT=$3
CAT_JSON=$4
DST=$5
DS=$6
# Finally run your job. Here's an example of a python script.

# python eval_coco.py $SLURM_TASK_ARRAY_ID
# python train_efusion_vkitti.py $SLURM_TASK_ARRAY_ID
python inference.py --model_name=$MODEL_NAME --batch_size=$BATCH_SIZE --checkpoint=$CHECKPOINT --categories_json=$CAT_JSON --dst=$DST --data_source=$DS --nodes=1 --ngpus=1 --ip_adress $ip1 $SLURM_TASK_ARRAY_ID