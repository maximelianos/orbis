import numpy as np
import torch
import torch.nn as nn

from networks.DiT.dit import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

class PositionEmbedding1d(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): Embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Positional embeddings will be created dynamically
        # Use register_buffer with persistent=False in __init__ to reserve the name
        self.register_buffer('pos_embed', None, persistent=False)
        self._pos_embed_shape = None  # Track current shape
    
    def forward(self, t):
        """
        Compute 1D sinusoidal positional embeddings from timestep values.

        Args:
            t: (B,) diffusion timesteps

        Returns:
            pos_embed: (B, embed_dim) positional embeddings
        """
        t_np = t.detach().reshape(-1).cpu().numpy().astype(np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, t_np)  # (B, embed_dim)
        pos_embed = torch.from_numpy(pos_embed).to(device=t.device, dtype=torch.float32)

        self.pos_embed = pos_embed
        self._pos_embed_shape = (t_np.shape[0],)
        return self.pos_embed

    
class PositionEmbedding2d(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): Embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Positional embeddings will be created dynamically
        # Use register_buffer with persistent=False in __init__ to reserve the name
        self.register_buffer('pos_embed_2d', None, persistent=False)
        self._pos_embed_2d_shape = None  # Track current shape
        
    def forward(self, height, width, device):
        """
        Get or create 2D sinusoidal positional embeddings.
        
        Args:
            height: Spatial height
            width: Spatial width
            device: Device to place embeddings on
            
        Returns:
            pos_embed: (H*W, embed_dim) positional embeddings
        """
        current_shape = (height, width)
        
        # Create new embeddings if shape changed or not yet created
        if self.pos_embed_2d is None or self._pos_embed_2d_shape != current_shape:
            # Generate 2D sincos positional embeddings
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                grid_size=[height, width],
                cls_token=False,
                extra_tokens=0
            )
            pos_embed = torch.from_numpy(pos_embed).float()  # (H*W, embed_dim)
            
            # Update buffer
            self.pos_embed = pos_embed.to(device)
            self._pos_embed_shape = current_shape
        
        return self.pos_embed