# DiT: the original architecture and the version in this repo

DiT ("Scalable Diffusion Models with Transformers", Peebles and Xie, ICCV 2023) replaces
the U-Net of latent diffusion with a plain ViT. The code we use lives in
[networks/DiT/dit.py](../networks/DiT/dit.py). Its `DiT` class is the original model with
video frames added, and `STDiT` / `SwinSTDiT` are factorized variants built on it. This
note goes through the original design first and then lists what our file changes.

## 1. The original model

### Data flow

```
noisy latent z_t  (B, C, I, I)          e.g. C=4, I=32 for 256x256 images (SD VAE)
   | PatchEmbed: conv, kernel = stride = p
   v
tokens (B, T, D),  T = (I/p)^2           + fixed 2D sin-cos position embedding
   |
   |   c = t_emb + y_emb  (B, D)         timestep and class label, summed
   v
N x DiTBlock(x, c)                       self-attention + MLP, adaLN-Zero on c
   v
FinalLayer(x, c)                         LayerNorm, adaLN shift/scale, Linear(D -> p*p*2C)
   v
unpatchify -> (B, 2C, I, I)              predicted noise and the variance (learn_sigma)
```

The model works only in the latent space of a frozen VAE. A 256x256 image becomes a
32x32x4 latent, and the transformer never sees pixels.

### Patchify

`PatchEmbed` from timm is one conv with kernel and stride `p`. Every p x p patch of the
latent becomes one token of width D. So the patch size sets the sequence length: p=2 on a
32x32 latent gives 256 tokens, p=8 gives only 16. The position embedding is the MAE 2D
sin-cos table (`get_2d_sincos_pos_embed`). It is stored as a parameter with
`requires_grad=False`, so it is never trained. Half of the channels encode the row and the
other half the column.

### Conditioning

There are two conditions, the diffusion step t and the ImageNet class y.

- `TimestepEmbedder`: sinusoidal features of t (256 dims, `[cos, sin]`, max period
  10000) and then `Linear -> SiLU -> Linear` to width D. In the paper t is an integer in
  0..999, so the frequencies are already spread out without rescaling.
- `LabelEmbedder` (not in our file): a lookup table with one extra "null" row. During
  training the label is replaced by the null class 10% of the time. That is what makes
  classifier-free guidance possible at sampling time.

The two embeddings are added into a single vector `c` of shape (B, D). This vector does
not go into the token sequence. It only drives the modulation inside every block.

The paper compares four ways to inject `c`:

1. **In-context**: append t and y as two extra tokens, like a CLS token.
2. **Cross-attention**: tokens attend to the condition tokens in an extra attention layer
   per block. About 15% more Gflops.
3. **adaLN**: replace the LayerNorm affine parameters by a shift and scale regressed from `c`.
4. **adaLN-Zero**: adaLN, plus a gate on each residual branch, with the regression layer
   initialized to zero.

adaLN-Zero gave the best FID and is also the cheapest, so every DiT model uses it.

### The DiT block (adaLN-Zero)

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(c).chunk(6)
x = x + gate_msa * Attn(modulate(LN(x), shift_msa, scale_msa))
x = x + gate_mlp * MLP (modulate(LN(x), shift_mlp, scale_mlp))
```

with `modulate(x, shift, scale) = x * (1 + scale) + shift`, and
`adaLN_modulation = SiLU -> Linear(D, 6D)`, one per block.

- The LayerNorms have no affine parameters of their own. The shift and scale come from `c`,
  so the normalization depends on the timestep and the class.
- The `1 + scale` form means that a zero output from the modulation layer leaves the
  normalized tokens unchanged.
- The gates are per channel and also come from `c`.
- The last Linear of `adaLN_modulation` starts at zero. Then every gate is 0 and every block
  is the identity at initialization, and the network starts as just patchify + final layer.
  This is the "Zero" part. It is the same trick as zero-initializing the last layer of a
  residual block (Goyal et al.), and it made training noticeably more stable.

The MLP is `Linear(D, 4D) -> GELU(tanh) -> Linear(4D, D)`. Attention is standard
multi-head self-attention over all T tokens.

### Final layer

`FinalLayer` is LayerNorm without affine, then an adaLN shift and scale from `c` (no gate),
then `Linear(D, p*p*out_channels)`. The output is reshaped back to the latent grid by
`unpatchify`. With `learn_sigma=True` the output has 2C channels: the noise prediction
and the interpolation coefficient of the variance (as in Improved DDPM). The Linear and the
modulation layer are both zero-initialized, so the untrained model outputs zeros.

### Initialization (`initialize_weights`)

- All Linears get Xavier uniform weights and zero biases.
- The patch embedding conv is initialized like a Linear (Xavier on the flattened kernel).
- The timestep MLP weights are normal with std 0.02.
- Every `adaLN_modulation[-1]` is set to zero, and so is the whole final layer.

### Model sizes

The configs follow ViT. The name is size/patch, e.g. DiT-XL/2.

| Model  | Layers N | Width D | Heads |
|--------|---------:|--------:|------:|
| DiT-S  | 12       | 384     | 6     |
| DiT-B  | 12       | 768     | 12    |
| DiT-L  | 24       | 1024    | 16    |
| DiT-XL | 28       | 1152    | 16    |

The main result of the paper is that FID tracks Gflops. Making the model deeper or wider
helps, and so does a smaller patch (more tokens). The parameter count alone predicts
quality poorly. DiT-XL/2 reached FID 2.27 on ImageNet 256x256 with guidance.

### Training setup in the paper

AdamW, learning rate 1e-4 constant, no weight decay, no warmup, batch 256, EMA of the
weights with decay 0.9999. Horizontal flips are the only augmentation. The diffusion part is
taken from ADM: 1000 steps with a linear beta schedule, epsilon prediction with the simple
MSE loss, plus the variational bound term for the learned variance. Sampling used 250 DDPM
steps.

## 2. What our `dit.py` changes

The default arguments of our `DiT` (`hidden_size=1152, depth=28, num_heads=16`) are
DiT-XL, but the model predicts video frames instead of images.

**Frames and context.** `forward(target, context, t, frame_rate)` takes the noisy target
frames (B, F_pred, C, H, W) and the clean context frames (B, F_ctx, C, H, W). They are
concatenated along time, each frame is patchified separately, and then all frames are
flattened into one sequence of (F_ctx + F_pred) * HW tokens. So `DiT` does full
spatio-temporal attention: every patch of every frame sees every other. After the blocks
only the last F_pred frames go through the final layer.

**Extra embeddings**, added to the tokens before the blocks:

- `frame_emb`: a learned (1, max_num_frames, 1, D) table, initialized with a truncated
  normal. It is indexed from the end (`max_num_frames - (F_target + F_ctx):`), so the last
  frame always gets the last embedding, whatever the context length.
- `frame_rate_encoder`: a fixed sin-cos encoding of the frame rate, normalized to
  [0, 1] over `frequency_range`. This makes the model aware of the time step between frames.

**No class label.** `get_condition_embeddings` returns only the timestep embedding, so
`c = t_emb`.

**Context regularization** in `preprocess_inputs`, training only:

- With probability `drop_ctx_rate` the context is dropped and the model predicts from the
  target alone. This is the video analogue of the null label for classifier-free guidance.
  The coin is flipped once per batch, not per sample.
- Otherwise, with probability `ctx_noise_aug_prob`, Gaussian noise with std
  `ctx_noise_aug_ratio` is added to the context frames of the samples whose t is at least
  that ratio. So the model learns not to trust its context exactly, which matters at rollout
  time when the context is its own earlier prediction.

**Attention details.** The block uses timm's `Attention` with `qk_norm=True` (LayerNorm on
queries and keys, which DiT did not have) and optional dropout. `learn_sigma` defaults to
False, because the models here are trained with flow matching and only predict a velocity.
`modulate` also accepts 4D tensors (B, F, N, D) for the factorized blocks.

### STDiT

`STDiT` swaps the blocks for `STDiTBlock`, which factorizes attention. Tokens stay in a
(B, F, N, D) layout, and each block does:

1. spatial attention inside each frame (`(b f) n d`), then a spatial MLP,
2. temporal attention across frames at each patch position (`(b n) f d`), then a temporal MLP.

The single modulation Linear now gives 9 vectors (shift, scale and gate for spatial
attention, spatial MLP and temporal MLP). By default the temporal attention itself is not
modulated: shift and scale are 0 and the gate is 1. `modulate_time_attn=True` adds a
separate zero-initialized modulation for it. `causal_time_attn=True` passes a lower
triangular mask, so a frame only attends to earlier frames. The cost drops from
O((F N)^2) for full attention to O(F N^2 + N F^2).

Note that `norm_time_attn` is the identity when the temporal attention is not modulated,
so in the default setting that branch has no pre-norm. And `norm4` is created but never used.

### SwinSTDiT and SwinSTDiTNoExtraMLP

`SwinSTDiTBlock` replaces the spatial attention by a Swin block (window attention, with
the windows shifted by half a window in every second layer). The temporal attention is
plain and unmodulated. The Swin block has its own MLP inside, so `SwinSTDiTNoExtraMLP`
keeps the `STDiTBlock` and only swaps its `space_attn` for a bare `SwinAttention`, which
avoids having two MLPs in a row.

## 3. Where DiT shows up in exp_navsim

The trajectory denoisers reuse parts of this design:

- `FinalLayer` is imported as it is and zero-initialized the same way.
- `CrossAttentionBlock` in [networks/blocks.py](../networks/blocks.py) follows the adaLN-Zero
  block, but with cross-attention instead of self-attention and 8 modulation vectors instead
  of 6, because the context tokens get their own shift and scale. AdaLN-Zero can be switched
  off there with `adaln=False`.
- The modulation Linear is not per block. One `AdaLNSingle` is shared by the whole stack, and
  each block adds a learned offset. This is the "adaLN-single" variant from PixArt-alpha,
  and with a small width it saves most of the modulation parameters.
- The diffusion time t is in [0, 1] for flow matching, not 0..999. So
  `exp_navsim.embeddings.sincos_embedding` multiplies t by 1000 first, to get the same
  frequency coverage as DiT's integer timesteps.
