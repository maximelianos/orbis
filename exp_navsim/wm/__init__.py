"""World-model-conditioned trajectory prediction.

    wm_runner.py  frozen stage-2 Orbis world model, run in latent space only
    denoiser.py   the (small) trajectory denoiser that consumes its output
    model.py      LightningModule gluing the two onto the flow-matching core
    config.yaml   training config (the training command is on top of it)

See exp_navsim/readme.md, section "World-model conditioning".
"""
