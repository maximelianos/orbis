"""
Spatial Aggregation Blocks

Contains blocks for aggregating spatial information from image latents.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .pos_emb import PositionEmbedding1d, PositionEmbedding2d

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import 2D positional embedding and AdaLN-Zero modulation from DiT
from networks.DiT.dit import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, modulate


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block between key-value tokens and query tokens.
    Supports AdaLN-Zero conditioning.
    
    Architecture:
        - Input tokens: (B, N, D) key-value tokens
        - Query tokens: (B, M, D)
        - Condition vector: (B, D)
        - Cross-attention: queries attend to key-value tokens
        - AdaLN conditioning: modulate with shift/scale/gate from timestep
        - Output: (B, M, D) aggregated tokens
    """
    
    def __init__(
        self,
        embed_dim,       # Embedding dimension (512 in Vaswani et al.)
        ff_dim,          # Hidden dimension of feed-forward (2048 in Vaswani et al.)
        num_heads=8,     # Number of attention heads
        dropout=0.1,     # Dropout rate
    ):
        """
        Initialize CrossAttentionBlock.
        
        Args:
            embed_dim: Channel dimension of input embeddings
            ff_dim: Hidden dimension for feed-forward layer
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        
        # K, V and Q embedding
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norms (without affine - already included in AdaLN)
        self.norm_input = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm_query = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm_output = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        
        # Multi-head cross-attention
        # Key/Value: input tokens. Aggregation by query tokens.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        # AdaLN modulation layers
        # Generate 8 parameters:
        # - shift, scale, gate for queries and output
        # - shift, scale for context
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 8 * embed_dim, bias=True)
        )
        # AdaLN-zero: initialize to zero for stability
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, context, query_tokens, condition):
        """
        Forward pass: aggregate tokens via cross-attention.
        
        Args:
            context: Input tokens of shape (B, N, D)
               where B=batch, N=sequence length, D=channels
            query_tokens: shape (B, M, D)
            condition: single vector (B, D) for AdaLN conditioning
        
        Returns:
            aggregated: Output tokens of shape (B, M, D)
        """
        B, N, D = context.shape
        _, M, _ = query_tokens.shape
        
        # Validate input dimensions
        assert D == self.embed_dim, f"Expected {self.embed_dim} channels, got {D}"
        assert condition.shape == (B, D), f"Expected ({B}, {D}), got {condition.shape}"
        
        # Get AdaLN modulation parameters: shift, scale, gate for queries, context and output
        shift_q, scale_q, gate_q, shift_c, scale_c, shift_o, scale_o, gate_o = self.adaLN_modulation(condition).chunk(8, dim=1)
        # Each: (B, D)
        
        # Layer normalization 
        context_norm = self.norm_input(context)  # (B, N, D)
        query_norm = self.norm_query(query_tokens)  # (B, M, D)
        
        # Apply AdaLN scale, shift to context
        context_norm = modulate(context_norm, shift_c, scale_c)
        query_norm = modulate(query_norm, shift_q, scale_q)  # (B, M, D)
        
        # Project to keys and values
        keys = self.k_proj(context_norm) # (B, N, D)
        values = self.v_proj(context_norm) # (B, N, D)
        query = self.query_proj(query_norm)  # (B, M, D)
        
        # Cross-attention: queries attend to all input tokens
        # query: (B, M, D)
        # key, value: (B, N, D)
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=keys,
            value=values,
        )
        # attn_out: (B, M, D)
        # attn_weights: (B, M, N)
        
        expected_shape = (B, M, D)
        assert attn_out.shape == expected_shape, f"Expected attn_out shape {expected_shape}, got {attn_out.shape}"

        # Attention residual branch: add queries to attention output
        attn_out = query_tokens + gate_q.unsqueeze(1) * attn_out

        # =========================== feed-forward
        
        # Normalize + modulate before feed-forward branch
        ff_in = self.norm_output(attn_out)
        ff_in = modulate(ff_in, shift_o, scale_o)
        
        # Pointwise feed-forward
        output = self.feed_forward(ff_in)  # (B, M, D)
        
        # Apply AdaLN modulation and gating to output
        output = gate_o.unsqueeze(1) * output  # (B, M, D)
        
        # Residual connection with attention branch
        output = output + attn_out  # (B, M, D)
        
        # Validate output shape
        expected_shape = (B, M, D)
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {output.shape}"
        
        return output


class SpatialBlock(nn.Module):
    """
    Spatial cross-attention block that aggregates image latents with query tokens.
    Query tokens are simply 1d positional embeddings.
    
    Extends CrossAttentionBlock by adding 2D positional embeddings to spatial tokens.
    No temporal modeling, uses cross-attention to aggregate spatial information
    from (H, W) grid into M query tokens.
    Supports AdaLN-Zero conditioning with condition vector.
    
    Architecture:
        - Input: (B*T, D, H, W) image latents, (B*T, M, D) query, (B, D) condition vector
        - Flatten spatial dims: (B*T, H*W, D)
        - Add 2D positional embeddings (no temporal embeddings)
        - Cross-attention aggregation with AdaLN conditioning
        - Output: (B*T, M, D) aggregated tokens
    """
    
    def __init__(
        self,
        embed_dim,       # Embedding dimension
        ff_dim,          # Hidden dimension for pointwise feed-forward layer
        num_predictions, # Number of tokens to predict (M)
        num_heads=8,     # Number of attention heads
        dropout=0.1,     # Dropout rate
    ):
        """
        Initialize SpatialBlock.
        
        Args:
            embed_dim: Channel dimension of input latents
            ff_dim: Hidden dimension for attention computation
            num_predictions: Number of tokens to predict (M)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_predictions = num_predictions
        
        # Cross-attention aggregation with AdaLN
        self.cross_attn_block = CrossAttentionBlock(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.pos_embed_1d = PositionEmbedding1d(embed_dim)
        self.pos_embed_2d = PositionEmbedding2d(embed_dim)
    
    def forward(self, x, condition):
        """
        Forward pass: aggregate spatial information via cross-attention.
        
        Args:
            x: Input latents of shape (B*T, D, H, W)
               where B=batch, T=frames, D=channels, H=height, W=width
            condition: shape (B*T, D) for AdaLN conditioning
               Note: should be repeated for each frame in the batch
        
        Returns:
            aggregated: Output tokens of shape (B*T, M, D)
        """
        BT, D, H, W = x.shape
        M = self.num_predictions
        
        # Validate input dimensions
        assert D == self.embed_dim, f"Expected {self.embed_dim} channels, got {D}"
        
        # Flatten spatial dimensions: (B*T, D, H, W) -> (B*T, H*W, D)
        x_flat = rearrange(x, 'bt c h w -> bt (h w) c')
        
        # Add positional embeddings (broadcast across batch)
        pos_embed = self.pos_embed_2d(H, W, x.device).unsqueeze(0)  # (1, H*W, embed_dim)
        x_pos = x_flat + pos_embed  # (B*T, H*W, embed_dim)
        
        # Generate queries
        pos_emb = self.pos_embed_1d(torch.arange(0, M, device=x.device)) # (M, D)
        query = pos_emb.view(1, M, D).expand(BT, M, D) # (BT, M, D)
        
        # Apply cross-attention aggregation with AdaLN conditioning
        aggregated = self.cross_attn_block(x_pos, query, condition)  # (B*T, M, D)
        
        # Validate output shape
        expected_shape = (BT, M, self.embed_dim)
        assert aggregated.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {aggregated.shape}"
        
        return aggregated


class TemporalBlock(nn.Module):
    """
    Temporal cross-attention block that predicts future tokens from context.
    
    Extends CrossAttentionBlock by adding 1D positional embeddings to temporal tokens.
    Uses cross-attention to predict M future tokens from N context tokens.
    Supports AdaLN conditioning.
    
    Architecture:
        - Input: (B, N, D) context tokens + (B, D) condition vector
        - Generate query: (1, M, D) timesteps [0, 1, ... M-1] encoded with positional embedding
        - Add 1D positional embeddings
        - Cross-attention prediction with AdaLN conditioning
        - Output: (B, M, D) predicted tokens
    """
    
    def __init__(
        self,
        embed_dim,       # Embedding dimension
        ff_dim,          # Hidden dimension for pointwise feed-forward layer
        num_heads=8,     # Number of attention heads
        dropout=0.1,     # Dropout rate
    ):
        """
        Initialize TemporalBlock.

        Args:
            embed_dim: Channel dimension of input embeddings
            ff_dim: Hidden dimension for pointwise feed-forward layer
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_input_length = 1000

        # Cross-attention prediction with AdaLN
        self.cross_attn_block = CrossAttentionBlock(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.pos_embed_1d = PositionEmbedding1d(embed_dim)

    def forward(self, context, query, condition):
        """
        Forward pass: predict future tokens from context via cross-attention.

        The number of predicted tokens M is taken from the query, so the same
        block works for any prediction length.

        Args:
            context: Input context tokens of shape (B, N, D)
               where B=batch, N=context length, C=channels
            query: (B, M, D) — one row per predicted token
            condition: AdaLN conditioning vector of shape (B, D)

        Returns:
            predictions: Output tokens of shape (B, M, D)
        """
        B, N, D = context.shape
        M = query.shape[1]
        assert D == self.embed_dim, f"Expected {self.embed_dim} channels, got {D}"

        # Cross-attention prediction with AdaLN conditioning
        predictions = self.cross_attn_block(context, query, condition)  # (B, M, embed_dim)

        expected_shape = (B, M, self.embed_dim)
        assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"

        return predictions

