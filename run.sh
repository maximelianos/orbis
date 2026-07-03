#!/bin/bash
# sbatch --gpus-per-node=1 /work/dlclarge1/velikanm-max/orbis/run.sh

# Define the partition on which the job shall run.
#SBATCH --partition lmbhiwidlc_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name orbis                    # short: -J <job name>
#SBATCH --time 23:00:00

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
conda activate orbis_env

# Running the job
start=`date +%s`

#python train_nuplan.py --config configs/nuplan.yaml
# 2025-12-19T14-39-41_nuplan 400 epochs
#  --resume logs_nuplan/2025-12-19T14-39-41_nuplan/checkpoints/last.ckpt
#  --resume logs_nuplan/2026-01-16T00-34-20_nuplan/checkpoints/last.ckpt (500 epochs)
#  --resume logs_nuplan/2026-01-17T20-37-40_nuplan/checkpoints/last.ckpt
#  2026-01-18T22-22-00_nuplan
# python train_nuplan.py --config configs/nuplan.yaml --resume logs_nuplan/2026-01-18T22-22-00_nuplan/checkpoints/last.ckpt
# python train_nuplan.py --config configs/nuplan.yaml

# python train_nuplan.py --config configs/game.yaml --max_steps 20000
# python evaluate/test_game.py --config configs/game.yaml --last_ckpt --logdir logs_nuplan

sleep 1000000000
python train_nuplan.py --config configs/nuplan_encoder.yaml --logdir logs_exp --max_steps 100000
#python train_nuplan.py --config configs/nuplan.yaml --logdir logs_exp --max_steps 200000
#python evaluate/test_nuplan.py --last_ckpt --logdir logs_nuplan --num_samples 100

#python train_nuplan.py --config evaluate/exp_game/game.yaml --max_steps 50000
#python train_nuplan.py --config configs/nuplan_encoder.yaml --logdir logs_exp --max_steps 5000

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
