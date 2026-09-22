"""Sin-cos embeddings shared by the trajectory denoisers.

Two kinds of scalar enter the denoisers, and they need different treatment:

  * the diffusion time t, a continuous value in [0, 1]. The sin-cos basis spans
    four decades of frequency, so scalars confined to [0, 1] leave every band but
    the highest almost constant and most of the embedding carries nothing. The
    input is therefore spread over [0, 1000) first, exactly as DiT's
    `TimestepEmbedder` does. `sincos_embedding` handles this case.

  * the frame index, an integer 0, 1, 2, ... Integers already span the basis, so
    no spread is applied. The embedding of frame i must not depend on the episode
    length (the old `arange(T) / T` made it depend on the longest episode in the
    batch, since batches are padded), so the table is indexed by the absolute
    frame number. `FrameIndexEmbedding` holds it.

Both are computed in torch on the input's device. `networks.pos_emb.PositionEmbedding1d`
round-trips through numpy on the CPU, which forces a synchronisation at every one
of the 50 sampling steps, and the table here is built once at construction rather
than at every forward pass (the DiT convention, see networks/DiT/dit.py).
"""

import torch
import torch.nn as nn


def sincos_embedding(values, dim, spread=1000.0):
    """Closed-form 1D sin-cos embedding of scalars.

    Args:
        values: (M,) scalars, normally in [0, 1].
        dim:    even embedding width.
        spread: factor applied before the basis. 1000 for values in [0, 1] (so
                the low-frequency bands actually vary); 1 for integer positions,
                which span the basis on their own.
    Returns:
        (M, dim) float32 on the same device as `values`.
    """
    assert dim % 2 == 0, f"embedding width must be even, got {dim}"
    half = dim // 2
    omega = torch.arange(half, device=values.device, dtype=torch.float32) / half
    omega = 1.0 / (10000.0 ** omega)                       # (half,)
    out = values.float().reshape(-1, 1) * spread * omega.reshape(1, -1)
    return torch.cat([out.sin(), out.cos()], dim=1)


class FrameIndexEmbedding(nn.Module):
    """Sin-cos embedding of the absolute frame index, tabulated once.

    The table is a non-persistent buffer: it never enters a checkpoint (it is
    fully determined by `dim`) but it follows the module across devices and
    dtypes, so a forward pass costs one slice instead of one sinusoid
    computation.
    """

    def __init__(self, dim, max_frames=1000):
        """
        Args:
            dim:        embedding width D.
            max_frames: longest episode the table can serve. NAVSIM episodes are
                        ~40 frames, so the default is far above what is needed
                        and costs max_frames * D floats.
        """
        super().__init__()
        index = torch.arange(max_frames, dtype=torch.float32)
        # spread=1: integer positions already cover the basis frequencies.
        self.register_buffer("table", sincos_embedding(index, dim, spread=1.0),
                             persistent=False)

    def forward(self, num_frames):
        """Embedding of frames 0 .. num_frames-1, as (num_frames, D)."""
        assert num_frames <= self.table.shape[0], (
            f"episode has {num_frames} frames but the frame table holds "
            f"{self.table.shape[0]}; raise max_frames")
        return self.table[:num_frames]
