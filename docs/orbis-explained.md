# Orbis: STDiT Module and World Model Detailed Explanation

This document provides a comprehensive explanation of the **STDiT (Spatial-Temporal DiT)** module and the **World Model (Stage 2)** implementation in the Orbis codebase, based on the paper "Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models".

---

## Table of Contents

1. [Overview](#overview)
2. [STDiT Module Architecture](#stdit-module-architecture)
3. [World Model (Stage 2) Implementation](#world-model-stage-2-implementation)
4. [Data Flow and Training Process](#data-flow-and-training-process)
5. [Key Design Decisions](#key-design-decisions)

---

## Overview

### Two-Stage Architecture

Orbis uses a **two-stage pipeline** for autonomous driving video prediction:

```
Stage 1 (Tokenizer):  Images → Continuous Encoder → Latent Tokens (frozen during Stage 2)
Stage 2 (World Model): Latent Tokens → STDiT-based Diffusion Model → Next Frame Prediction
```

- **Stage 1**: A hybrid tokenizer (encoder) compresses images into compact latent representations
- **Stage 2**: A flow-matching based diffusion model (using STDiT architecture) predicts the next frame in latent space

---

## STDiT Module Architecture

**Location**: `networks/DiT/dit.py`

### What is STDiT?

**STDiT** stands for **Spatial-Temporal Diffusion Transformer**. It extends the standard DiT (Diffusion Transformer) architecture to handle **spatio-temporal data** (video sequences) by decomposing attention into:
1. **Spatial Attention**: Captures relationships within each frame
2. **Temporal Attention**: Captures relationships across frames

### Core STDiTBlock Implementation

The fundamental building block is `STDiTBlock` (lines 235-303 in `dit.py`):

```python
class STDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Decomposes into spatial and temporal attention paths.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0,
                 causal_time_attn=False, modulate_time_attn=False, **block_kwargs):
        super().__init__()

        # SPATIAL PATH COMPONENTS
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = Attention(hidden_size, num_heads=num_heads, ...)  # Spatial self-attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_mlp = Mlp(...)  # Spatial MLP

        # TEMPORAL PATH COMPONENTS
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, ...)  # Temporal self-attention
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.time_mlp = Mlp(...)  # Temporal MLP

        # CONDITIONING (AdaLN modulation)
        # Generates 9 conditioning parameters: 3 for spatial attention, 3 for spatial MLP, 3 for temporal MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
```

#### Key Architectural Features:

1. **Dual Path Processing**: Spatial and temporal operations are **separate and sequential**
2. **Adaptive Layer Normalization (AdaLN)**: Conditioning signal modulates the layer norms through shift, scale, and gate parameters
3. **Optional Causal Temporal Attention**: Can use causal masking for temporal attention to respect temporal ordering

### Forward Pass Through STDiTBlock

```python
def forward(self, x, c):
    """
    Args:
        x: Input tensor of shape [B, F, N, D]
           B = batch size
           F = number of frames
           N = number of spatial tokens (patches)
           D = hidden dimension
        c: Conditioning signal [B, D] (timestep embedding from diffusion process)
    """
    B, F, N, D = x.shape

    # Step 1: Generate 9 conditioning parameters from timestep embedding
    (shift_msa, scale_msa, gate_msa,        # Spatial attention modulation
     shift_mlp_s, scale_mlp_s, gate_mlp_s,  # Spatial MLP modulation
     shift_mlp_t, scale_mlp_t, gate_mlp_t)  = self.adaLN_modulation(c).chunk(9, dim=1)

    # ========== SPATIAL ATTENTION PATH ==========
    # Step 2: Apply spatial attention (operates on each frame independently)
    x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)  # Modulate normalization
    x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d')  # Flatten batch and frames
    x_ = self.space_attn(x_modulated)  # Self-attention across spatial tokens
    x_ = rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)  # Restore structure
    x = x + gate_msa.unsqueeze(1).unsqueeze(1) * x_  # Gated residual connection

    # Step 3: Spatial MLP
    x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
    x = x + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.space_mlp(x_modulated)

    # ========== TEMPORAL ATTENTION PATH ==========
    # Step 4: Apply temporal attention (operates across frames for each spatial position)
    if self.modulate_time_attn:
        shift_mta, scale_mta, gate_mta = self.adaLN_time_attn_modulation(c).chunk(3, dim=1)
    else:
        shift_mta, scale_mta, gate_mta = torch.zeros_like(...), torch.zeros_like(...), torch.ones_like(...)

    x_modulated = modulate(self.norm_time_attn(x), shift_mta, scale_mta)
    x_modulated = rearrange(x_modulated, 'b f n d -> (b n) f d')  # Group by spatial position

    # Optional causal masking for temporal attention
    time_attn_mask = torch.tril(torch.ones(F, F)) if self.causal_time_attn else None
    x_ = self.time_attn(x_modulated, attn_mask=time_attn_mask)  # Self-attention across time
    x_ = rearrange(x_, '(b n) f d -> b f n d', b=B, n=N, f=F)
    x = x + gate_mta.unsqueeze(1).unsqueeze(1) * x_  # Gated residual

    # Step 5: Temporal MLP
    x_modulated = modulate(self.norm3(x), shift_mlp_t, scale_mlp_t)
    x = x + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(x_modulated)

    return x
```

#### Why This Design?

1. **Factorized Spatio-Temporal Attention**: Processing spatial and temporal dimensions separately is more **efficient** than full 3D attention and allows the model to specialize each path
2. **AdaLN Modulation**: The conditioning signal (timestep `t` in diffusion) controls the strength of each operation, crucial for diffusion models
3. **Residual Connections with Gates**: Learned gating allows the model to adaptively route information

### STDiT Full Model

The complete `STDiT` class (lines 575-605) extends the base `DiT` class:

```python
class STDiT(DiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32,
                 hidden_size=1152, depth=28, num_heads=16, ...):
        super().__init__(...)

        # Replace standard DiT blocks with STDiT blocks
        self.blocks = nn.ModuleList([
            STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                      dropout_rate=dropout, causal_time_attn=causal_time_attn,
                      modulate_time_attn=modulate_time_attn)
            for _ in range(depth)  # Typically depth=24
        ])
```

**Key Components Inherited from DiT**:

1. **Patch Embedding** (`x_embedder`): Converts images to patches
2. **Positional Encoding** (`pos_embed`): 2D sinusoidal positional embeddings
3. **Timestep Embedding** (`t_embedder`): Embeds diffusion timestep `t`
4. **Frame Embeddings** (`frame_emb`): Learnable embeddings for each frame position
5. **Frame Rate Encoder** (`frame_rate_encoder`): Encodes the video frame rate (important for generalization)

### STDiT Forward Pass

```python
def forward(self, target, context, t, frame_rate, return_features=False):
    """
    Args:
        target: [B, F_pred, C, H, W] - Noisy frames to denoise
        context: [B, F_ctx, C, H, W] - Context frames (past observations)
        t: [B] - Diffusion timestep
        frame_rate: [B] - Video frame rate

    Returns:
        out: [B, F_pred, C, H, W] - Predicted velocity field (for flow matching)
    """
    f_pred = target.size(1)

    # Step 1: Get timestep conditioning
    c = self.get_condition_embeddings(t)  # [B, D]

    # Step 2: Preprocess inputs (patch embedding + positional encoding + frame embeddings)
    x = self.preprocess_inputs(target, context, t, frame_rate)  # [B, F_total, N, D]

    # Step 3: Apply STDiT blocks sequentially
    for i, block in enumerate(self.blocks):
        x = block(x, c)  # Each block processes all frames together

    # Step 4: Final layer to predict output (velocity for flow matching)
    out = self.final_layer(x[:, -f_pred:], c)  # Only take predicted frames
    out = self.postprocess_outputs(out)  # Unpatchify to image shape

    return out
```

### SwinSTDiT Variant

For **high-resolution** inputs (512×288), Orbis uses `SwinSTDiT` (lines 608-613):

```python
class SwinSTDiT(STDiT):
    def __init__(self, ...):
        super().__init__(...)
        # Replace spatial attention with Swin Transformer blocks for efficiency
        self.blocks = nn.ModuleList([
            SwinSTDiTBlock(...)  # Uses windowed attention instead of full spatial attention
            for layer_idx in range(depth)
        ])
```

**Why Swin Transformer?**
- **Windowed Attention**: Reduces computational complexity from O(N²) to O(N) for spatial dimension
- **Shifted Windows**: Enables cross-window connections while maintaining efficiency

---

## World Model (Stage 2) Implementation

**Location**: `models/second_stage/fm_model.py`

The world model is implemented as a **Flow Matching** based diffusion model that operates in the latent space produced by the tokenizer (Stage 1).

### Flow Matching Overview

Instead of the standard DDPM diffusion process, Orbis uses **Flow Matching** (Lipman et al., 2022), which:
- Defines a continuous trajectory from data distribution to noise distribution
- Uses an ODE-based sampling process (deterministic)
- More stable and efficient than score-based diffusion

### Model Class Architecture

```python
class Model(pl.LightningModule):
    """
    Base model class for the flow matching version of the world model.

    Key responsibilities:
    1. Encode images to latent space using frozen tokenizer
    2. Train STDiT to predict velocity fields in latent space
    3. Generate new frames through ODE integration
    4. Support autoregressive rollout for long sequences
    """
    def __init__(self, *, tokenizer_config, generator_config,
                 sigma_min=1e-5, timescale=1.0, enc_scale=4, ...):
        super().__init__()

        # The denoising backbone (STDiT)
        self.vit = self.build_generator(generator_config)

        # The frozen tokenizer from Stage 1
        self.ae = self.build_tokenizer(tokenizer_config)

        # EMA (Exponential Moving Average) model for better generation
        self.ema_vit = init_ema_model(self.vit)

        # Flow matching parameters
        self.sigma_min = sigma_min  # Minimum noise level (default: 1e-5)
        self.timescale = timescale  # Time scaling factor
        self.enc_scale = enc_scale  # Scaling for latent codes
```

### Flow Matching Formulation

The flow matching objective defines a trajectory from data to noise:

```python
def alpha(self, t):
    """Weight for data: 1 - t"""
    return 1.0 - t

def sigma(self, t):
    """Weight for noise: sigma_min + t * (1 - sigma_min)"""
    return self.sigma_min + t * (1.0 - self.sigma_min)

def A(self, t):
    """Coefficient for data in velocity target"""
    return 1.0

def B(self, t):
    """Coefficient for noise in velocity target"""
    return -(1.0 - self.sigma_min)
```

**Interpretation**:
- At t=0: Pure data (x₀)
- At t=1: Pure noise (ε)
- Intermediate: x_t = (1-t)·x₀ + σ(t)·ε

The model learns to predict the **velocity field**: v(x_t; t) = -dx_t/dt

### Training Process

```python
def training_step(self, batch, batch_idx):
    """
    Training step for flow matching objective.

    Process:
    1. Encode images to latent space
    2. Sample random timestep t
    3. Add noise to target frame according to flow matching schedule
    4. Predict velocity with STDiT
    5. Compute MSE loss against ground truth velocity
    """
    images, frame_rate = self.get_input(batch, 'images')

    # Step 1: Encode to latent space using frozen tokenizer
    # x: [B, F, C, H, W] where F = number of frames
    x = self.encode_frames(images)  # Uses self.ae.encode()

    b, f, e, h, w = x.size()

    # Step 2: Split into context (past frames) and target (frame to predict)
    if f == 1:
        context = None
        target = x.squeeze(1)
    else:
        context = x[:, :-1].clone()  # All frames except last
        target = x[:, -1]            # Last frame

    # Step 3: Sample random timestep t ~ Uniform(0, 1)
    t = torch.rand((x.shape[0],), device=x.device)

    # Step 4: Add noise according to flow matching schedule
    # x_t = (1-t)·target + σ(t)·ε
    target_t, noise = self.add_noise(target, t)
    target_t = target_t.unsqueeze(1)  # Add frame dimension

    # Step 5: Predict velocity with STDiT
    # pred: [B, 1, C, H, W] - predicted velocity field
    pred = self.vit(target_t, context, t, frame_rate=frame_rate)

    # Step 6: Compute ground truth velocity
    # v = dx_t/dt = A(t)·x + B(t)·ε
    target_velocity = self.A(t) * target + self.B(t) * noise

    # Step 7: MSE loss between predicted and true velocity
    loss = ((pred.float() - target_velocity.float()) ** 2).mean()

    self.log("train/loss", loss, prog_bar=True, logger=True, ...)

    return loss
```

### Key Training Details

1. **Frozen Tokenizer**: The autoencoder (`self.ae`) is frozen - only the STDiT (`self.vit`) is trained
2. **Context Dropout**: During training, context frames are randomly dropped 50% of the time (configured in STDiT config)
3. **Context Noise Augmentation**: When context is present, noise is added 50% of the time to improve robustness
4. **EMA Updates**: After each batch, the EMA model is updated for better generation quality

```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    """Update EMA model after each batch"""
    update_ema(self.ema_vit, self.vit, decay=0.9999)
```

### Sampling Process (Inference)

The `sample()` method generates a single next frame:

```python
@torch.no_grad()
def sample(self, images=None, latent=False, eta=0.0, NFE=20,
           sample_with_ema=True, num_samples=8, frame_rate=None):
    """
    Generate a single next frame using ODE integration.

    Args:
        images: Context frames [B, F_ctx, C, H, W]
        NFE: Number of Function Evaluations (integration steps)
        eta: Stochasticity parameter (0 = deterministic ODE, >0 adds noise)
        sample_with_ema: Use EMA model (recommended for generation)

    Returns:
        target_t: Generated latent frame [B, C, H, W]
        images: Decoded RGB frame [B, 1, 3, H, W]
    """
    net = self.ema_vit if sample_with_ema else self.vit
    device = next(net.parameters()).device

    # Step 1: Encode context frames (if provided)
    if images is not None:
        if not latent:
            context = self.encode_frames(images)
        else:
            context = images.clone()
    else:
        context = None

    # Step 2: Initialize target with pure noise (t=1)
    input_h, input_w = self.vit.input_size[0], self.vit.input_size[1]
    target_t = torch.randn(num_samples, 1, self.vit.in_channels,
                          input_h, input_w, device=device)

    # Step 3: ODE integration from t=1 to t=0
    t_steps = torch.linspace(1, 0, NFE + 1, device=device)  # [1.0, 0.95, ..., 0.0]

    for i in range(NFE):
        t = t_steps[i].repeat(target_t.shape[0])

        # Predict velocity at current timestep
        neg_v = net(target_t, context, t=t * self.timescale, frame_rate=frame_rate)

        dt = t_steps[i] - t_steps[i+1]  # Time step size

        # ODE update: x_{t-dt} = x_t + v(x_t, t) * dt
        # Optional stochastic term (eta > 0)
        dw = torch.randn_like(target_t) * torch.sqrt(dt)
        diffusion = dt
        target_t = target_t + neg_v * dt + eta * torch.sqrt(2 * diffusion) * dw

    # Step 4: Decode final latent to RGB image
    last_frame = target_t.clone()
    images = self.decode_frames(last_frame)

    return target_t.squeeze(1), images
```

**ODE Integration Details**:
- Starts from pure noise (t=1)
- Iteratively follows the learned velocity field
- Each step: x_{t-dt} = x_t + v(x_t,t)·dt
- After NFE steps (typically 20-30), reaches clean sample (t=0)

### Autoregressive Rollout

For long video generation, frames are generated **autoregressively**:

```python
def roll_out(self, x_0, num_gen_frames=25, latent_input=True,
             eta=0.0, NFE=20, sample_with_ema=True, num_samples=8):
    """
    Generate a long video sequence autoregressively.

    Process:
    1. Start with initial context frames (x_0)
    2. Generate next frame
    3. Append generated frame to context, remove oldest frame (sliding window)
    4. Repeat for num_gen_frames

    Args:
        x_0: Initial context frames [B, F_ctx, C, H, W]
        num_gen_frames: Number of frames to generate

    Returns:
        x_all: All frames (context + generated) [B, F_ctx+num_gen_frames, C, H, W]
        samples: Decoded RGB frames [B, num_gen_frames, 3, H, W]
    """
    b, f = x_0.size(0), x_0.size(1)

    # Encode initial context
    if latent_input:
        x_c = x_0.clone()
    else:
        x_c = self.encode_frames(x_0)

    x_all = x_c.clone()
    samples = []

    # Autoregressive generation loop
    for idx in tqdm(range(num_gen_frames), desc="Rolling out frames"):
        # Generate next frame given current context
        x_last, sample = self.sample(images=x_c, latent=True, eta=eta,
                                     NFE=NFE, sample_with_ema=sample_with_ema,
                                     num_samples=num_samples)

        # Append to full sequence
        x_all = torch.cat([x_all, x_last.unsqueeze(1)], dim=1)

        # Sliding window: drop oldest frame, add newest
        x_c = torch.cat([x_c[:, 1:], x_last.unsqueeze(1)], dim=1)

        samples.append(sample)

    samples = torch.cat(samples, dim=1)
    return x_all, samples
```

**Sliding Window Context**:
- Context size is **fixed** (e.g., 5 frames)
- As new frames are generated, oldest frames are dropped
- This maintains constant memory and computational cost

### ModelIF: Factorized Token Variant

`ModelIF` (lines 351-481) extends the base `Model` for **factorized tokenization**:

```python
class ModelIF(Model):
    """
    Flow matching model for token factorization latents.

    The tokenizer produces TWO types of latents:
    1. Detail tokens (reconstruction-focused)
    2. Semantic tokens (DINO-distilled, semantic-focused)

    These are concatenated along the channel dimension before being fed to STDiT.
    """
    def __init__(self, ..., enc_scale=1.89066, enc_scale_dino=3.45062, ...):
        super().__init__(...)
        self.enc_scale_dino = enc_scale_dino  # Separate scaling for semantic tokens

    @torch.no_grad()
    def encode_frames(self, images):
        """Encode frames with factorized tokenization"""
        x = self.ae.encode(images)["continuous"]  # Returns dict with two token types
        x0 = x[0] * self.enc_scale        # Detail tokens
        x1 = x[1] * self.enc_scale_dino   # Semantic tokens (DINO)
        x = torch.cat([x0, x1], dim=1)    # Concatenate along channel
        return x

    def training_step(self, batch, batch_idx):
        """Training with separate loss tracking for detail and semantic"""
        ...
        loss = ((pred.float() - target.float()) ** 2)

        # Split loss into detail and semantic components
        loss_recon = loss[:, :loss.size(1)//2].mean()  # Detail tokens
        loss_sem = loss[:, loss.size(1)//2:].mean()    # Semantic tokens
        loss = loss.mean()

        # Log separately for analysis
        self.log("train/loss_recon", loss_recon, ...)
        self.log("train/loss_sem", loss_sem, ...)

        return loss
```

**Why Factorized Tokens?**
- **Detail tokens**: Capture fine-grained visual information (textures, edges)
- **Semantic tokens**: Capture high-level semantic content (objects, scene structure)
- DINO distillation provides **semantic structure** that helps with long-horizon prediction
- Separate scaling allows balancing between reconstruction quality and semantic consistency

---

## Data Flow and Training Process

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

Input: Video frames [B, 6, 3, 256, 256] at 5Hz
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1: TOKENIZER (Frozen)                                          │
│  ┌─────────────┐        ┌──────────────┐                            │
│  │ Image       │───────▶│ Encoder      │────▶ Latent [B,6,C,16,16]  │
│  │ [B,6,3,H,W] │        │ (ViT-based)  │     (C=16 or 32)           │
│  └─────────────┘        └──────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
    Split: Context [B,5,C,16,16] + Target [B,1,C,16,16]
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2: FLOW MATCHING WORLD MODEL                                   │
│                                                                       │
│  1. Sample t ~ Uniform(0,1)                                          │
│  2. Add noise: x_t = (1-t)·target + σ(t)·ε                          │
│  3. STDiT prediction:                                                │
│     ┌────────────────────────────────────────────────┐              │
│     │ Input: [target_t, context, t, frame_rate]      │              │
│     │   ↓                                             │              │
│     │ Patch Embed + Pos Embed + Frame Embed          │              │
│     │   ↓                                             │              │
│     │ STDiTBlock 1 (spatial + temporal attention)    │              │
│     │   ↓                                             │              │
│     │ STDiTBlock 2                                    │              │
│     │   ...                                           │              │
│     │ STDiTBlock 24                                   │              │
│     │   ↓                                             │              │
│     │ Final Layer → Velocity prediction              │              │
│     └────────────────────────────────────────────────┘              │
│  4. Loss: MSE(predicted_velocity, true_velocity)                    │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
    Backprop through STDiT only (tokenizer frozen)
```

### Inference (Rollout)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

Input: Context frames [B, 5, 3, 256, 256]
           │
           ▼
    Encode to latent [B, 5, C, 16, 16]
           │
           ▼
    ┌────────────────────────────────────────┐
    │ AUTOREGRESSIVE GENERATION LOOP         │
    │                                        │
    │  For i in range(num_frames_to_gen):   │
    │    1. Initialize: x_T ~ N(0,I)        │
    │    2. ODE Integration (t=1→0):        │
    │       for t in [1.0, 0.95, ..., 0]:   │
    │         v = STDiT(x_t, context, t)    │
    │         x_{t-dt} = x_t + v·dt         │
    │    3. Decode: RGB = Decoder(x_0)      │
    │    4. Update context:                 │
    │       context = [context[1:], x_0]    │
    └────────────────────────────────────────┘
           │
           ▼
    Output: Generated video [B, num_frames, 3, 256, 256]
```

---

## Key Design Decisions

### 1. Why STDiT Instead of Standard DiT?

**Problem**: Video data has spatio-temporal structure that standard spatial-only attention cannot capture.

**Solution**: STDiT factorizes attention into:
- **Spatial Attention**: Models relationships within each frame (objects, scene structure)
- **Temporal Attention**: Models motion and dynamics across frames

**Benefits**:
- More parameter efficient than full 3D attention
- Each path can specialize (spatial for appearance, temporal for motion)
- Proven effective in video generation (used in Latte, OpenSora, etc.)

### 2. Why Flow Matching Instead of DDPM?

**Flow Matching Advantages**:
- **Deterministic sampling** (when eta=0): More stable, predictable generation
- **Fewer sampling steps**: Typically 20-30 vs 1000 for DDPM
- **Continuous trajectories**: Smoother interpolation between noise and data
- **Better training stability**: Simpler objective (predict velocity)

**Comparison**:
```
DDPM:         predict ε(x_t, t)  [predict noise]
Flow Matching: predict v(x_t, t)  [predict velocity]
```

### 3. Why Frozen Tokenizer?

**Rationale**:
1. **Two-stage training is more stable**: Tokenizer convergence doesn't affect world model
2. **Memory efficient**: No gradients through encoder during Stage 2
3. **Modular design**: Can improve tokenizer independently
4. **Faster training**: Only train STDiT (469M params) not full pipeline

### 4. Why Context Dropout and Noise Augmentation?

From `dit.py` (lines 523-532):

```python
if self.training:
    # Drop context 50% of the time
    if torch.rand(1) < self.drop_ctx_rate:
        context = None
    # Add noise to context 50% of the time
    elif torch.rand(1) < self.ctx_noise_aug_prob:
        mask = (t >= self.ctx_noise_aug_ratio)
        aug_noise = torch.randn_like(context)
        context[mask] = context[mask] + aug_noise[mask] * self.ctx_noise_aug_ratio
```

**Purpose**:
1. **Context dropout**: Prevents over-reliance on context, improves unconditional generation
2. **Noise augmentation**: Makes model robust to noisy/corrupted context (important for autoregressive rollout where errors accumulate)

### 5. Why EMA (Exponential Moving Average)?

```python
def update_ema(ema_model, model, decay=0.9999):
    """
    EMA smooths parameter updates: θ_ema = decay·θ_ema + (1-decay)·θ
    """
    for name, param in model.parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
```

**Benefits**:
- **Smoother parameters**: Reduces noise in parameter space
- **Better generation quality**: EMA model typically generates higher quality samples
- **Standard practice**: Used in all modern diffusion models (DDPM, LDM, etc.)

### 6. Configuration Details

From `configs/stage2.yaml`:

```yaml
generator_config:
  target: networks.DiT.dit.STDiT
  params:
    max_num_frames: 64          # Maximum context window
    hidden_size: 768            # Model dimension
    depth: 24                   # Number of STDiT blocks
    num_heads: 16               # Attention heads (768/16 = 48 dim per head)
    mlp_ratio: 4                # MLP hidden dim = 4×768 = 3072
    input_size: [16, 16]        # Latent spatial size
    patch_size: 1               # No further patching (already in latent space)
    in_channels: 16             # Latent channels (or 32 for factorized)
    dropout: 0.0
    ctx_noise_aug_ratio: 0.1    # Noise level for context augmentation
    ctx_noise_aug_prob: 0.5     # Probability of adding noise
    drop_ctx_rate: 0.1          # Probability of dropping context entirely
```

**Model Size**:
- Parameters: 469M (much smaller than Vista/GEM which have ~1B+ params)
- Efficient: Only 24 layers vs 28-32 in larger models

---

## Summary

### STDiT Module
- **Core Innovation**: Factorized spatio-temporal attention with adaptive layer norm conditioning
- **Architecture**: 24 STDiTBlock layers, each with spatial and temporal paths
- **Conditioning**: Timestep embeddings modulate attention and MLPs via AdaLN
- **Variants**: Standard STDiT for 256×256, SwinSTDiT for 512×288

### World Model (Stage 2)
- **Paradigm**: Flow matching-based diffusion in latent space
- **Training**: Predict velocity fields from noisy latents
- **Inference**: ODE integration from noise to clean samples (20-30 steps)
- **Autoregressive**: Sliding window context for long rollouts (>20 seconds)
- **Efficiency**: Frozen tokenizer, EMA for stability, context augmentation for robustness

### Key Insights from Paper
1. **Continuous > Discrete**: Flow matching outperforms discrete MaskGIT approach
2. **Tokenizer matters less for continuous**: Flow matching is robust to tokenizer design choices
3. **Long-horizon success**: Unlike Vista/GEM, Orbis maintains quality over 20+ seconds
4. **Simplicity wins**: No depth supervision, no BEV, just raw video → latents → generation

This architecture achieves **state-of-the-art** long-horizon video prediction with only **280 hours** of training data and **469M parameters**, demonstrating the power of the flow matching + STDiT combination.
