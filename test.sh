#!/bin/bash
# sbatch --gpus-per-node=1 /work/dlclarge1/velikanm-max/orbis/test.sh

# Define the partition on which the job shall run.
#SBATCH --partition lmbhiwidlc_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name orbis-test                    # short: -J <job name>
#SBATCH --time 01:00:00

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH --nodes 1

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A-HelloCluster.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-HelloCluster.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 40GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
echo $HOME
. ~/.bashrc
conda activate orbis_env

# Running the job
start=`date +%s`

# Test the trained model
# Update the checkpoint path to your trained model
# 2026-01-17T20-37-40_nuplan  # 700 epochs
python evaluate/test_nuplan.py --config configs/nuplan.yaml --ckpt logs_nuplan/2026-01-17T20-37-40_nuplan/checkpoints/last.ckpt --num_steps 20 --seed 42

# python evaluate/test_nuplan.py --config configs/nuplan.yaml --ckpt logs_nuplan/2026-01-23T04-35-06_nuplan/checkpoints/last.ckpt --num_steps 20 --seed 42

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
