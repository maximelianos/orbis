"""Data layer for exp_navsim: raw-episode loading, latent caching and the
fixed-length window datasets used for training.

Modules (mirroring the config's data_long / cache / data_latent subkeys):
  * navsim_long    — NavsimLongDataset (one whole episode per index)
  * cache_latents  — encode every episode and cache per-frame latents to HDF5
  * latent_dataset — NavsimLatentDataset / NavsimRawWindowDataset training windows
  * buffer         — BufferDataset in-memory ring buffer wrapper
"""
