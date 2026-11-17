Work only in the project folder /scratch/local/velikanov/work/orbis. Use subagents.

You can run commands.
You cannot run on the GPU directly. You need to connect to tmux: "tmux -S /tmp/tmux-socket-12345 attach -t mysession".  Run "env CUDA_VISIBLE_DEVICES=0 bash" inside tmux at first to access the gpu.

Use python only through a miniconda env "navsim". You cannot create new envs.
Run commands through tmux only.
Make sure your scripts never consume more than 40 GB CPU RAM in total.
Make sure you don't run out of disk space.
Run at most 3 concurrent runs not to overload the CPUs.
Don't run in background - run in separate tmux windows so I can see the progress.
Reuse tmux windows - keep 4 windows open at most.

## Persistence
The session may end abruptly at any time. Therefore always write intermediate results in the README of respective output/
and read it when starting up.

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

## Evaluation speed

In intermediate experiments, evaluate the method on a subset of the whole test set. Evaluate only the final method on the whole test set.

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

# Trajectory diversity

Continue research on trajectory diversity. Come up with new learning pipelines that would show the diverse trajectories without collapsing. Think, use websearch, and justify the ideas before starting to implement. Plot with BEV, not your simplistic trajectory plots that you had until now. The pipeline shouldn't be too hand-crafted, but it must guarantee diverse predictions. Make the prediction easier and faster for the model to learn - for example, regress the current speed from an ORBIS block and condition the diffusion model on it.

# Diffusion plus regression

Why regression performs better than diffusion:
* The data is mostly unimodal given conditioning
* Diffusion is a harder task to learn than regression

Try following existing approaches to facilitate diffusing the trajectory:

* Residual diffusion. Use regression as the base and diffuse only the residual.
* Sample N, then rerank. Use a learned scorer or a PDM-like cost, or a Hydra-MDP-style vocabulary with scoring heads. This is where generative proposals typically gain over regression on EPDMS. Pure sampling gives poor trajectory candidates.
* Stabilize. Use EMA

Creating multi-modal ground truth:
* Supervise with simulator or PDM-based scores over a trajectory vocabulary or anchors (Hydra-MDP style). Many trajectories get labeled as good, so you obtain mode information the log never gives.
* Fine-tune with RL or reward-weighted objectives on EPDMS (e.g., GRPO-style on diffusion/flow), so the model is pushed toward any high-scoring mode rather than the logged one.
* Augment data with perturbed ego states or counterfactual scenes.

Normalization and prediction target:
* Trajectory normalization and noise scale: poses spanning tens of meters need standardization, or the high-noise end of the schedule is meaningless.
* Prediction target (x0 vs epsilon vs v): for low-dimensional trajectories, predicting x0 is often better.

Regression being better and faster here is consistent with the literature. Diffusion’s edge shows up mainly with strongly multimodal data, closed-loop compounding errors, guidance, or a selection stage in the end.

Try out all of these approaches and compare them in a table and visualizations.

## More diffusion

1. Train diffusion supervised by PDM-based scores over a trajectory vocabulary or anchors (Hydra-MDP style). Many trajectories get labeled as good, so you obtain mode information the log never gives.

2. Normalization and prediction target:
* Trajectory normalization and noise scale: poses spanning tens of meters need standardization, or the high-noise end of the schedule is meaningless.
* Prediction target (x0 vs epsilon vs v): for low-dimensional trajectories, predicting x0 is often better.
Train with different nomralizations, compare them visually and quantitatively.

## Proper driving metrics

Compute the driving metrics on NAVSIM v2 similarly to this page (see readme there): https://github.com/kesai-labs/py123d_garage
to be able to compare results faithfully.

## More compositional generalization

Look at the compositional generalization study in report/papers. Do deep research about compositional generalization approaches. Think about better experiments on compositional generalization and implement them.

## Artifact

Generate a website to present the results. It should be scientific, but compact, easy to read,
only the most important information and experiment results.
* Graph plots. Add PNG plots for most important visualizations. Apply good-looking styles to the plots.
* Trajectory plots.
* Videos. Create videos with the front view + BEV view with GT trajectory in green and predicted trajectory / samples in red. Draw 3D trajectory in the video frame for GT and predicted trajectory by projecting 3D to 2D. Show videos at 2 frames per second. Show only the most interesting scenes: cross-roads, failures, successes. Show around 3 most intersting videos. If the video doesn't feature something worth noting, don't show it.
* Front frame + BEV pictures. Add most intersting pictures for method ablation.
* Tables. Add numbers from existing literature and KE:SAI and highlight that they are existing.

## Website correction

Title of project: Testing the internal representation of world models with an attentive diffusion probe for ego vehicle trajectory prediction

Add citations of previous works in whole text.

Introduction: list the main findings, don't put the numbers on blocks.
Add a table to show number of scenes in the dataset, not text.
In section titles, remove the gray "second title".
Add conclusion about which training is better - diffusion or direct prediction.

What are inputs, outputs to the models? Length of context, length of prediction.
What are the metrics used: energy, dispersion, etc? Why are these metrics used and not others?
"Ego status and command alone reach energy 2.39" - add to plot, explain why so good,
explain how you incorporate this input for later sections.

Remove the claim that less euler steps lead to fake collapse: with the large enough model and long training, any amount of steps lead to collapse, but keep the plot.
What means calibrated?
Add plot for 92k scenes.

How does fast planner without command perform? How long does the diffusion probe train?
"planner backbone, flow matching matches regression" - reformulate more clearly.
"Reward-weighted fine-tuning" - explain.
Check if TransFuser-K1 uses LIDAR BEV. Correct the "inputs" column.

Add section about experiments with different types of normalization, add plots.

Add section about different sampling architectures: anchors, k means, nucleous. Explain these approaches with SVG schemes.

Instead of "published", write author and year of paper.

"Three scenes worth watching" -> Video examples.
Why do the videos jump? Predict for the whole episode length.

Add section for trajectory representation:
* Representing trajectories as polynomials
* Polynomial of order 2, 3 is better in val than order 6 or raw poses
* Poly repr does not prevent memorisation

## Improving EPDMS

Continue improving the EPDMS on navsim v2.2, use proxy with small amount of scenes to speed up comparisons. Integrate our backbone with best approaches from existing literature from KE:SAIhttps://github.com/kesai-labs/py123d_garage . Speed up the training, but leave an option to produce diverse trajectories. Visualize your results in pictures with front view and BEV. You may use additional inputs from data.
---

Continue improving epdms.

1. Experiment with input features, use heuristics. Download ORBIS 2, try its different blocks.
2. Train in a dream: rollout ORBIS 2 given several selected trajectories, observe latent space in future and use these features.
3. Use additional cameras: extract their latents too.
4. Use BEV encoding.

Use deep research to read the top methods from KESAI readme, think about how to adopt their approaches, and implement.

## Improving EPDMS 08.11

Look at the EODMS table in NAVSIM v2 here https://github.com/kesai-labs/py123d_garage , take the best methods (TerraTransfer++, DriveZero-Scale,
DrivoR, etc.), read the details of these methods, use the parts of these methods that gave them the most advantage,
incorporate them in our method, and achieve at least 57 EPDMS.
