"""Batch collation for variable-length episodes.

With `num_frames=0` in the config each sample carries a whole episode, and
episodes differ in length, so the default PyTorch collate (which stacks tensors
and therefore requires identical shapes) no longer applies.

`pad_collate` pads every time-major tensor with zeros up to the longest episode
in the batch, so the batch is a rectangular (B, T_max, ...) tensor as before. The
true per-sample length travels along in

    batch["metadata"]["length"]   (B,) int64

and the model turns it into a frame mask (see NavsimTrajectoryModel.frame_mask):
the padded tail is still *predicted* — every sample gets the same T_max output —
but it is excluded from the loss and from the validation metrics.

Padding applies to every entry of the sample dict independently, so it covers
both the trajectory/velocity (T, 2) and the image latents (T_img, C, H, W), whose
own length differs when a short episode has fewer frames than `context_images`.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def _collate_values(values):
    """Collate the list of per-sample values of ONE key of the sample dict.

    Args:
        values: list of length B holding sample[key] of each sample in the batch.
    Returns:
        A batched value: tensors are zero-padded along dim 0 (time) and stacked
        into (B, T_max, ...), nested dicts (e.g. "metadata") are collated
        key-by-key, python scalars become a (B,) tensor, and anything else is
        left as a plain list.
    """
    first = values[0]

    # Nested dict (e.g. {"length": int}) -> collate each of its keys separately.
    if isinstance(first, dict):
        return {k: _collate_values([v[k] for v in values]) for k in first}

    if isinstance(first, torch.Tensor):
        # Scalar tensors have no time axis to pad: plain stack -> (B,).
        if first.dim() == 0:
            return torch.stack(values, 0)
        # pad_sequence zero-pads dim 0 up to the longest entry and stacks the
        # result into (B, T_max, ...). Trailing dims must already match, which
        # they do (C, H, W of the latents / the 2 of the trajectory). When all
        # lengths are equal (fixed num_frames) this is just a stack.
        return pad_sequence(values, batch_first=True)

    # frame_rate and friends: python numbers -> (B,) tensor, as the default
    # collate would produce.
    if isinstance(first, (int, float, bool)):
        return torch.as_tensor(values)

    # Strings (tokens) and arbitrary objects: no sensible batching, keep the list.
    return values


def pad_collate(samples):
    """DataLoader collate_fn: batch episode dicts, padding to the longest one.

    Args:
        samples: list of per-sample dicts, all sharing the same keys.
    Returns:
        dict with the same keys, batched along a new leading dim of size B.
    """
    return {k: _collate_values([s[k] for s in samples]) for k in samples[0]}
