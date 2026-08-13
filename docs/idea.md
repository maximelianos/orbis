Config: configs/nuplan_encoder.py

Dataloader: nuplan_dataloader.NuPlanVelocityBuffered
* Depends on NuPlanVelocityDataset



Model: whole_context.DiffusionModel
* Depends on loaders_for_projects.Encoder

Visualization
* PDF with images and trajecotries: draw_nuplan.py

# To implement

(done, not tested) networks/norm.py: Implement TrajectoryNorm class, which is a pytorch module. It takes a batch, extracts batch["velocity"], and uses EMA to update the mean and variance of the velocity. See NuplanHDF for how the normalization works (in our case, it's pytorch tensors). Save the mean (dx, dy) and variance (dx, dy) as a nn parameter. Use the new class to normalize and denormalize the velocity in whole_context.py

(will not be done) evaluate/test_nuplan_with_encode.py: use the new dataloader and model to test. Implement similarly to test_nuplan.py

# Cache latents efficiently

Caching on the fly has its benefits:
* No need to run precomputation separately
* If data format changes (add world model latents), no need to re-run precomputation

To efficiently use saved latents between runs, I need:
* **No shuffling of train samples** - to go over same samples each time, load them from cache

Implement a dataloader loaders_for_projects/nuplan_long.py that loads whole episodes from the dataset.


Dataloader NuPlanVelocityDataset
* Depends on custom_multiframe_odo.MultiHDF5DatasetMultiFrameIdxMappingOdometry
* Which depends on custom_multiframe.MultiHDF5DatasetMultiFrameIdxMapping

DONE
Add option whole_episodes in class MultiHDF5DatasetMultiFrameIdxMapping(Dataset),
that tells to sample only the subvideo starting from the first frame and drop all others. Implement loaders_for_projects/nuplan_long.py, which has a dataloader similar
to NuPlanVelocityDataset, inherited from MultiHDF5DatasetMultiFrameIdxMappingOdometry.
This dataloader will have whole_episodes=True from the config, so it returns whole episodes
in getitem. It has a field with a sorted list of paths to the HDF files containing the episodes.
It returns "metadata" in the getitem dict:
batch["metadata"] = dict("path": "/path/to/hdf.h5")
Don't change the custom_* files apart from the option, only overload necessary functions in the new dataloader.
All other functionality is the same like in the original classes and NuPlanVelocityDataset.
Write a test, that uses draw_nuplan, to visualize several episodes.
Make minimal changes, but write comments like in nuplan_dataloader.
Create configs/long_data.yaml based on nuplan_export_config.

# 03.07.2026

# Dataloader
Implement dataloader for navsim. I want to load navsim-hard from
navsim/download/navhard_two_stage .
The dataloader must be concise. Use similar return dict like in nuplan_long. Also return velocity.
The dataloader should have config options like in long_data:
omit num_frames,
use stored_data_frame_rate,
use frame_rate,
omit aug.

This is a "long" dataloader - i.e. it loads full / long episodes from the data.

Implement a drawing function that visualizes a birds-eye-view, 5 evenly spaced camera observations, and the trajectory.
The function receives a batch dict and plots one of the samples. It plots depending on which fields are present in the batch.
Implement drawing in a separate file.

Visualize the long dataloader - first few episodes. Save one PDF. Print on the first page the statistics of the dataset -
how many episodes, distribution of lengths of episodes. Implement a separate run script for that.

## Cache for latents

Next, create a script to run the image encoder and cache image latents. Save the cache in data/ directory.
Keep the class for loading the encoder, encoding and decoding in separate file (like in encoder.py)
Enumerate episodes from the original long dataloader, and for each episode save an HDF file with per-frame latent representations.

Create a dataloader for the encoded latents. Reuse functions from the original long dataloader to write as little code as possible.
Keep the dataloader in a separate file. 
Implement a separate script to test decoding: it loads first several episodes and plots birds-eye-view (BEV),
decoded images, and trajectory using the episode visualization function you already implemented.

## Buffer

Use dataloader buffer implementation from brain_matching. The dataloader itself is totally different there,
but the buffer is useful.

# Model training

Use a config flag: load raw images or encoded. If raw, then you must load the encoder in the model class and encode
the latents. If encoded, then you use the dataloader for cached encodings and the model class does NOT load the encoder.

Predict the velocity of the vehicle (already impelemented in whole_context, you can mostly copy it).
GT trajectories must be normalized with EMA (see norm.py). Implement normalization in a separate file.

Use the same existing train.py script.

# Test metrics

There is almost no testing during current training. Implement a separation of episodes into train / validation split in the dataloader
(just a list of the episodes with in/out check + add a config flag).
In validation, run inference on same context several times (config flag - how many times),
measure MSE with the GT trajectory, but also STD of total turn. Measure total turn as the sum of angles of trajectory.
Implement metrics in a separate file.

Print only val loss in cmd logging, but save val/mse and val/std in tensorboard. Produce PDF with a small amount of test episodes
(run the same test episodes each time), where you visualize the predicted trajectories. This is already implemented in my code somewhere.

Create a test script to produce the PDF.

# General suggestions
You can import from any other files, but rewrite the dataloader and model, use train.py as main entry point.
Implement everything in folder "exp_navsim". Mostly, all this functionality is implemented in my code somewhere,
but you need to structure it well.

In navsim/ you can find NAVSIM simulator and data. In brain_matching is another project, copy the buffer dataset implementation
from there.

Make code concise, create more files but with separated functionality, so it's easier to read.
Describe your approach in exp_navsim/readme.

Keep configs for everything in one config file, but in that file use different subkeys to separate the logic.

Don't run any commands.

# Adding navsim-navtrain

I downloaded the navtrain subset of navsim in navsim/download directory. Write scripts in
exp_navsim/data to
1) load episodes from navtrain
2) cache latents
3) load latents
like I did for navsim-hard.
Use parts of existing data code as much as possible. If the existing code is not reusable,
try to decompose it more in order to be reusable.

Change the config to train on navtrain subset. Leave the option to use navhard.
Make code concise, create more files but with separated functionality, so it's easier to read.
Describe your approach in exp_navsim/readme.

Don't run any commands.

### Navhard and navtrain format

Frame format is identical between the two datasets — same keys, same structure. The differences are purely: (1) directory layout, (2) episode length (navhard = 5-frame scenes, navtrain logs = hundreds of frames), and (3) sensor root. Let me check a few more things: scene_tokens per navtrain log, and whether sensor blobs are fully present (download looked partial).

### Remove num predictions

num_predictions doesn't play any role in the blocks.py - TemporalBlock. Remove it.
In exp_navsim model, remove the restriction on episode length, leave only the context size. Just load the episode as it is, and predict all steps that are not in context. You'll need to change the model.

# 30.07.2026

Project aim: prove that the Orbis VAE + world model have learned representations that
contain information about ego-motion and trajectory planning.
Check if an attentive probe of a small size can decode the future trajectory from the WM latents.
Using diffusion, recover a distribution of future trajectories.
I will use nuReasoning dataset to train and test the attentive probe.

Check the nuReasoning website and tell
1) what's the duration of clips
2) how many clips, how is the dataset structured
3) do the clips have associated 2D trajectories

https://huggingface.co/datasets/qixuewei/nuReasoning

Suggest 3 experiments that I have not done so far (read report). Write in docs/future-plan.md.