Work only in the project folder /scratch/local/velikanov/work/orbis.

You can run commands.
You cannot run on the GPU directly. You need to connect to tmux: "tmux -S /tmp/tmux-socket-12345 attach -t mysession".  Run "env CUDA_VISIBLE_DEVICES=0 bash" inside tmux at first to access the gpu.

Use python only through a miniconda env "navsim". You cannot create new envs.
Run commands through tmux only.
Make sure your scripts never consume more than 20 GB RAM.
Make sure you don't run out of disk space.
Run at most 3 concurrent runs not to overload the CPUs.
Don't run in background - run in separate tmux windows so I can see the progress.
Reuse tmux windows - keep 4 windows open at most.

## Loop pipeline

Run in a loop:
1. When selecting a method to implement, do websearch to find accurate information.
2. Implement the method. Run on the data, save output in outputs/<nn_experiment_name> directory.
3. Save detailed summary and visualizations in outputs/<nn_experiment_name>
4. Run NN training, look at resulting tensorboard PDF with loss curves.

Document only major progress in docs/log.md, in the format "[month-day] one line of description". For more detailed description, create readmes in outputs/ folder.

Training runs still go to logs_exp/. Add readmes to logs_exp/<run> folders to remember what changes were applied for the run.

## Documentation

If you write something in docs/ folder, use the format yy-mm-dd-<doc_name>.md.

## Goal

The project goal is to obtain insights about the ORBIS world model internal feature space.
In the report/ the existing experiments are documented.

Current problem: We want to show, that a diffusion model trained to predict future trajectory can predict
a multimodal distribution of trajectories on the test, but the more the model trains, the more unimodal the distribution is.

My proposals:
* Encode the trajectory as coefficients of a polynom to prevent overfitting
* Decrease model capacity
* Compare latents from DINO, v Jepa, Orbis encoder q_sem, q_rec, Orbis WM.
* Compare training on latents from different blocks of world model (4 different blocks). Make a graph of performance vs block index.
* Decrease planning horizon according to navsim protocol.

* Compute EPDMS and its components, compare with other planners
* Because we have a trajectory diffusion model, measure per-episoded best and average trajectory.
* std of distribution for all encoders

* Check generalization: navsim train -> val (read navsim documentation how to do it properly!)
* (technical) cache world model latents for faster training

* Check compositional generalization
* better evaluation metric?

Do these experiments. Find other interesting experiments to perform. I need at least 3 other interesting results,
in the model efficiency, model analysis, encoder choice, or other findings.
Use websearch, perform the experiments according to the research loop above.
You will work autonomously from now.

Resructure the existing code into folders logically (but don't touch original orbis code). Remove unnecessary code,
tidy it up.


