"""
Spatial Aggregation Blocks

Contains blocks for aggregating spatial information from image latents.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import 2D positional embedding and AdaLN-Zero modulation from DiT
from networks.DiT.dit import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class AdaLNSingle(nn.Module):
    """
    AdaLN-Zero modulation computed once and shared by a stack of blocks
    (PixArt-alpha's "adaLN-single").

    Maps a condition vector (B, cond_dim) to the 6 modulation vectors of a
    ContextTransformer, flattened to (B, 6 * D). Each block adds its own learned
    (6 * D,) offset, which plays the role of the per-block bias, so the Linear
    here has none. The weight is zero-initialized: together with the zero
    offsets every gate starts at 0 and every block starts as the identity.
    """

    def __init__(self, cond_dim, embed_dim):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * embed_dim, bias=False),
        )
        nn.init.constant_(self.modulation[-1].weight, 0)

    def forward(self, condition):
        """(B, cond_dim) -> (B, 6 * D)."""
        return self.modulation(condition)


class ContextTransformer(nn.Module):
    """
    Pre-norm cross-attention block: query tokens (B, M, D) attend to context
    tokens (B, N, D), then a feed-forward, both residual.

    The context goes through an affine LayerNorm and a Linear K/V projection,
    the query through its norm and a Linear Q projection. Attention is SDPA,
    followed by a Linear output projection.

    adaln=True:  AdaLN-Zero on the query side. `modulation` (B, 6 * D) from the
                 caller's AdaLNSingle plus this block's zero-init offset gives
                 shift/scale for the query and the feed-forward input, and the
                 two gates.
    adaln=False: plain pre-norm block (affine LayerNorms, no gates);
                 `modulation` is ignored.
    """

    def __init__(self, embed_dim, ff_dim, num_heads, dropout, adaln=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.adaln = adaln
        self.norm_input = nn.LayerNorm(embed_dim, eps=1e-6)
        # With AdaLN the affine part of these norms comes from the modulation.
        self.norm_query = nn.LayerNorm(embed_dim, elementwise_affine=not adaln, eps=1e-6)
        self.norm_output = nn.LayerNorm(embed_dim, elementwise_affine=not adaln, eps=1e-6)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        if adaln:
            self.adaLN_offset = nn.Parameter(torch.zeros(6 * embed_dim))

    def forward(self, context, query_tokens, modulation=None, key_padding_mask=None):
        """
        Args:
            context:          (B, N, D) key/value tokens.
            query_tokens:     (B, M, D).
            modulation:       (B, 6 * D) output of an AdaLNSingle; only used with adaln.
            key_padding_mask: optional (B, N) bool, True = context token to ignore.
        Returns:
            (B, M, D)
        """
        B, N, D = context.shape
        M, H = query_tokens.shape[1], self.num_heads
        assert D == self.embed_dim and D % H == 0, f"Expected {self.embed_dim} channels, got {D}"

        q = self.norm_query(query_tokens)
        if self.adaln:
            assert modulation.shape == (B, 6 * D), f"Expected ({B}, {6 * D}), got {modulation.shape}"
            shift_q, scale_q, gate_q, shift_o, scale_o, gate_o = (
                (modulation + self.adaLN_offset).unsqueeze(1).chunk(6, dim=-1))   # each (B, 1, D)
            q = q * (1 + scale_q) + shift_q
        else:
            gate_q = gate_o = 1.0

        # SDPA takes True = attend, the opposite of key_padding_mask.
        attn_mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]  # (B, 1, 1, N)
        c = self.norm_input(context)
        q = self.q_proj(q)                                          # (B, M, D)
        k, v = self.kv_proj(c).chunk(2, dim=-1)                     # (B, N, D) each

        # Split heads: (B, L, D) -> (B, H, L, D/H).
        q = q.view(B, M, H, D // H).transpose(1, 2)
        k = k.view(B, N, H, D // H).transpose(1, 2)
        v = v.view(B, N, H, D // H).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = self.out_proj(attn.transpose(1, 2).reshape(B, M, D))  # merge heads, (B, M, D)
        x = query_tokens + gate_q * attn

        h = self.norm_output(x)
        if self.adaln:
            h = h * (1 + scale_o) + shift_o
        return x + gate_o * self.feed_forward(h)


class SpatialBlock(nn.Module):
    """
    Pools image latents (B*T, D, H, W) into M tokens (B*T, M, D): 
    T frames, M fixed 1D sin-cos query tokens
    cross-attend to the H*W grid cells, which carry a 2D
    sin-cos position embedding. Unconditioned - no AdaLN.
    """

    def __init__(self, embed_dim, ff_dim, num_predictions, num_heads, dropout):
        super().__init__()
        self.transformer = ContextTransformer(embed_dim, ff_dim, num_heads, dropout, adaln=False)
        # Query table built once; non-persistent, as it is fully determined by (M, D).
        query_pos = get_1d_sincos_pos_embed_from_grid(
            embed_dim, np.arange(num_predictions, dtype=np.float32))     # (M, D)
        self.register_buffer("query_pos", torch.from_numpy(query_pos).float(), persistent=False)
        # 2D table of the (H, W) grid, built on the first forward and rebuilt
        # only when the grid size changes.
        self.register_buffer("grid_pos", None, persistent=False)
        self.grid_size = None

    def forward(self, x):
        """x: (B*T, D, H, W) -> (B*T, M, D)."""
        BT, D, H, W = x.shape
        if self.grid_size != (H, W):
            grid_pos = get_2d_sincos_pos_embed(D, [H, W])                    # (H*W, D)
            self.grid_pos = torch.from_numpy(grid_pos).float().to(x.device)
            self.grid_size = (H, W)
        x = rearrange(x, 'bt c h w -> bt (h w) c') + self.grid_pos           # (B*T, H*W, D)
        return self.transformer(x, self.query_pos.expand(BT, -1, -1))
