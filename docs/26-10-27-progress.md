# Progress 2026-10-27 (superseded)

All queues below finished by 2026-10-29. Current state and results:
docs/26-10-29-results-summary.md.

Plan and protocol: docs/26-10-27-research-plan.md. One-line log: docs/log.md.

## Done

- **New NAVSIM-protocol pipeline** in exp_navsim/ (4 history CAM_F0 frames at 2 Hz in,
  8 poses over 4 s out). Index of 102,056 navtrain scenes (old pipeline: 9,577
  episodes). Validation holds out whole logs (98 logs, 9,675 scenes), since navtrain
  scenes overlap. See exp_navsim/README.md.
- **Restructure**: exp_navsim/ split into data/ encoders/ models/ eval/ viz/ configs/;
  the old whole-episode probe moved to exp_navsim/episode/ (still imports). Loose
  top-level scripts moved to tools/, scripts/slurm/, legacy/; two `*_trash.py` and
  a notebook deleted. Upstream Orbis folders untouched. All modules import.
- **Test data**: OpenScene test logs + test CAM_F0 downloaded to
  /scratch/global/velikanm/navsim/download/test_* (for navtest and navhard stage 1).
- **Feature caches** (data/features/navtrain, 9 x 16 grid): q_sem, q_rec (16 ch),
  dinov2 (64 PCA ch, 73.7% variance).
- **02 collapse reproduced** (outputs/02_collapse): q_sem probe on 10k scenes, val
  loss minimum at ~40k of 100k steps, val dispersion 0.96 -> 0.87, best-of-32
  minADE 0.686 -> 0.706 while mean-sample ADE improves. With 92k scenes the drift is
  much slower. Sample figure shows wide fans early, single (sometimes wrong) bundle late.
- **03 baselines** (outputs/03_baselines): constant velocity ADE 2.76, nearest
  ego-status lookup ADE 1.26 (ego status is a strong shortcut).
- **13 linear probes** (outputs/13_linear_probes/linear_probe.csv): DINOv2-64 >
  q_sem > q_rec on speed, progress, lateral, heading, command. At equal width
  (16 channels) DINOv2 and q_sem are nearly identical.

## Running in tmux (session mysession)

| window | queue file | content |
|---|---|---|
| orbis | outputs/04_data_scaling/queue.txt | data scaling n=2.5k/5k/20k/46k, capacity (06), DINOv2-16ch and history 1/2 (12) |
| orbis-2 | outputs/07_encoders/queue.txt | dinov2, q_rec, q_sem+q_rec, none, status, status+cmd, codec (05), compositional (10), no-Singapore (09) |
| orbis-dl | outputs/01_features/queue.txt | V-JEPA 2 + world-model caches (navtrain), navtest/navhard index and features, metric caches, then 07_vjepa2 and 08 world-model block runs |
| orbis-3 | - | short analysis jobs |

Each queue runs `tools/tmux/queue.sh <file>`; restarting it skips commands whose
log ends with `__EXIT 0`. Runs go to logs_exp/<name>/ (README.md with the note).
Note: orbis-2 has a stray plot command in its input buffer that runs when its
queue ends (harmless).

## Open issues / next steps

- navtest / navhard index build via SceneLoader is untested (first run is in the
  orbis-dl queue after the world-model cache).
- After runs finish: `exp_navsim.eval.scaling` for 04, `plot_runs` for 05/06/07/08,
  `evaluate --split navtest --save-predictions` + `eval.pdm score` for EPDMS (11),
  `evaluate --steps 1,2,4,...` for the Euler-step efficiency study (12),
  `evaluate --split holdout_val --holdout sg-one-north:left,us-ma-boston:right`
  for the compositional comparison, `--cities sg-one-north` for 09.
- Write READMEs for 04-13 and update the report section.
- An earlier V-JEPA 2 caching run was SIGKILLed at 30% (cause unknown); caching now
  writes outputs without memmap and stays well under 20 GB per process.
