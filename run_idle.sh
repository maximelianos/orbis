#!/bin/bash
# srun -p gpu --time 1:00:00 --gpus-per-node=1 --pty bash
# sbatch --gpus-per-node=1 /scratch/local/velikanov/work/orbis/run_idle.sh

# Define the partition on which the job shall run.
#SBATCH --partition gpu     # short: -p lmbhiwidlc_gpu-rtx2080

# Define a name for your job
#SBATCH --job-name brain                    # short: -J <job name>
#SBATCH --time 48:00:00

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH --nodes 1

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A-HelloCluster.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-HelloCluster.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 20GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
#source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
#conda activate hello_cluster_envc
echo $HOME
. ~/.bashrc
#conda activate torch

# Running the job
start=`date +%s`

#----Run tmux
tmux -S /tmp/tmux-socket-12345 new-session -d -s mysession
sleep 1000000000

# tmux -S /tmp/tmux-socket-12345 attach -t mysession
# env CUDA_VISIBLE_DEVICES=0 bash

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
