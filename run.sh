#!/bin/bash
# srun -p gpu --time 1:00:00 --gpus-per-node=1 --pty bash
# sbatch --gpus-per-node=1 /scratch/local/velikanov/work/orbis/run.sh

# Define the partition on which the job shall run.
#SBATCH --partition gpu    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name orbis                    # short: -J <job name>
#SBATCH --time 48:00:00

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH --nodes 1

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out

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


#---NAVSIM
#python -m exp_navsim.data.cache_latents --config exp_navsim/config.yaml
# python train_nuplan.py -c exp_navsim/config.yaml --max_steps 1000000 --logdir logs_navsim
# python tensorboard_to_pdf.py --logdir ./logs_navsim --last --name navtrain.pdf --from 1000

python -m exp_navsim.test_decode       --config exp_navsim/config.yaml --num 3
python train_nuplan.py -c exp_navsim/config.yaml --max_steps 200000 --logdir logs_navsim
python -m exp_navsim.test_model --config exp_navsim/config.yaml \
    --ckpt logs_navsim/2026-07-22T23-01-41_config/checkpoints/last.ckpt --num 6
python tensorboard_to_pdf.py --logdir ./logs_navsim --last --name navtrain.pdf --from 0


# sleep 1000000000
# python train_nuplan.py --config configs/nuplan_encoder.yaml --logdir logs_exp --max_steps 100000
#python train_nuplan.py --config configs/nuplan.yaml --logdir logs_exp --max_steps 200000
#python evaluate/test_nuplan.py --last_ckpt --logdir logs_nuplan --num_samples 100

#python train_nuplan.py --config evaluate/exp_game/game.yaml --max_steps 50000
#python train_nuplan.py --config configs/nuplan_encoder.yaml --logdir logs_exp --max_steps 5000

# --- nuReasoning

# cd /dsk/scratch/velikanm
# hf download qixuewei/nuReasoning   --repo-type dataset   --local-dir ./nuReasoning   --max-workers 2

cd /scratch/local/velikanov/work/orbis/nuReasoning
mkdir -p data_unzipped

find data -name "*.zip" | while read -r zip_file; do
    rel_path="${zip_file#data/}"
    rel_dir="$(dirname "$rel_path")"
    clip_name="$(basename "$zip_file" .zip)"

    out_dir="data_unzipped/${rel_dir}/${clip_name}"
    mkdir -p "$out_dir"

    unzip -n "$zip_file" -d "$out_dir"
done

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
