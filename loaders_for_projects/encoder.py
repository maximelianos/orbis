import sys
from pathlib import Path
import os
import torch
from omegaconf import OmegaConf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util import instantiate_from_config

class Encoder:
    def __init__(self, exp_dir, ckpt, config, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        exp_dir = Path(exp_dir).resolve()
        ckpt_path = (exp_dir / ckpt).resolve()
        config_path = (exp_dir / config).resolve()
        
        base_cfg = OmegaConf.load(config_path)
        
        model = instantiate_from_config(base_cfg.model)
        state = torch.load(str(ckpt_path), map_location="cpu")["state_dict"]
        model.load_state_dict(state, strict=True)
        model = model.to(self.device).eval()
        
        if hasattr(model, 'encode'):
            self.tokenizer = model
        elif hasattr(model, 'ae'):
            self.tokenizer = model.ae
        else:
            raise AttributeError("Model doesn't have 'encode' method or 'ae' attribute")

    @torch.inference_mode()
    def encode(self, x):
        """
        Encodes a batch of images into continuous latents.
        
        Args:
            x (torch.Tensor): Output from dataloader, typically (B, C, H, W).
            
        Returns:
            tuple: continuous_latents (q_rec, q_sem), each a 4D tensor (e.g. B, C_lat, H_lat, W_lat).
        """
        assert x.dim() == 4, f"Expected input of shape (B, C, H, W), got {x.shape}"
        with torch.autocast(dtype=torch.float16, device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            encoded = self.tokenizer.encode(x.to(self.device))
            continuous_latents = encoded["continuous"]
            
            assert isinstance(continuous_latents, tuple) and len(continuous_latents) == 2, \
                f"Expected continuous_latents to be a tuple of length 2, got {type(continuous_latents)}"
            assert continuous_latents[0].dim() == 4 and continuous_latents[1].dim() == 4, \
                f"Expected latents to be 4D tensors, got {continuous_latents[0].shape} and {continuous_latents[1].shape}"
                
            return continuous_latents

    @torch.inference_mode()
    def decode(self, continuous_latents):
        """
        Decodes continuous latents back into images.
        
        Args:
            continuous_latents (tuple): Tuple of (q_rec, q_sem) latents.
            
        Returns:
            torch.Tensor: Decoded images of shape (B, C, H, W).
        """
        assert isinstance(continuous_latents, tuple) and len(continuous_latents) == 2, \
            "Expected continuous_latents to be a tuple of length 2 (q_rec, q_sem)"
        assert continuous_latents[0].dim() == 4 and continuous_latents[1].dim() == 4, "Expected latents to be 4D tensors"
        
        with torch.autocast(dtype=torch.float16, device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            decoded, _ = self.tokenizer.decode(continuous_latents)
            assert decoded.dim() == 4, f"Expected decoded output of shape (B, C, H, W), got {decoded.shape}"
            
            return decoded
