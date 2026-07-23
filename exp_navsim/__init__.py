"""exp_navsim: self-contained NAVSIM long-episode trajectory-prediction pipeline.

See exp_navsim/readme.md for the overall design. Everything here is driven by a
single config (exp_navsim/config.yaml) whose subkeys separate the logic
(data_long / cache / data_latent / model / val / viz).
"""
