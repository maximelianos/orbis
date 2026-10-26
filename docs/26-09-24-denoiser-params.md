# Where the denoiser's ~400K parameters come from

Setting: `exp_navsim/config.yaml` as of 2026-09-24 and the code in
`exp_navsim/denoiser.py` / `networks/blocks.py`:

- `hidden_dim` D = 64, 8 heads, feed-forward width 2D = 128
- `image_encoder_channels` = 16, two latents concatenated (q_sem + q_rec), so 32 input channels
- `image_tokens` N = 64, `context_traj` = 0, `context_images` = 2
- **4** `TemporalBlock`s in `TrajectoryDenoiser.temporal_stack`

Exact total: **438,978 trainable** parameters (Lightning prints "439 K"),
plus **5 non-trainable** in `TrajectoryNorm` (mean 2, var 2, initialized flag 1).
The tokenizer is not counted. It is only built lazily when `encode_images: true`.

## Top level

| Module | Params | Share |
|---|---:|---:|
| `temporal_stack` (4 × TemporalBlock) | 315,904 | 72.0% |
| `image_aggregator` | 81,088 | 18.5% |
| `context_mlp` | 12,544 | 2.9% |
| `pred_mlp` | 12,544 | 2.9% |
| `fm_time_mlp` | 8,320 | 1.9% |
| `frame_mlp` | 8,320 | 1.9% |
| `output_norm` + `output_mlp` | 258 | 0.1% |
| **Total** | **438,978** | |

## One attention block (78,976)

`TemporalBlock` and `SpatialBlock` are both a thin wrapper around
`CrossAttentionBlock`, so they have the same count. Position embeddings and
query tables are buffers, not parameters.

| Part | Shape | Params |
|---|---|---:|
| `adaLN_modulation` | Linear(64 → 8·64) | 33,280 |
| `cross_attn` (nn.MultiheadAttention) | in_proj 3·(64×64) + out_proj 64×64, with biases | 16,640 |
| `feed_forward` | Linear(64 → 128) + Linear(128 → 64) | 16,576 |
| `k_proj`, `v_proj`, `query_proj` | 3 × Linear(64 → 64) | 12,480 |
| 3 LayerNorms | `elementwise_affine=False` | 0 |
| **Block total** | | **78,976** |

In terms of D this is 19·D² weights + 18·D biases (8D² adaLN, 4D² MHA,
4D² feed-forward, 3D² extra projections).

## The rest

- `image_aggregator` = `image_enc_mlp` Linear(32 → 64) = 2,112, plus one `SpatialBlock` = 78,976.
- `context_mlp`, `pred_mlp`: Linear(2 + 2·64 = 130 → 64) = 8,384, plus Linear(64 → 64) = 4,160.
- `fm_time_mlp`, `frame_mlp`: 2 × Linear(64 → 64) = 8,320 each.
- `output_norm` LayerNorm(64) = 128, `output_mlp` Linear(64 → 2) = 130.

Buffers that hold no parameters: `frame_pos_emb.table` (1000 × 64),
`token_pos_emb.table` (64 × 64), `SpatialBlock.query_pos` (64 × 64), and the
1D/2D sin-cos position embeddings.

## Things that stand out

1. **AdaLN takes 38% of the model.** 5 blocks × 33,280 = 166,400. Each block
   regresses 8 modulation vectors of width D from the time/frame condition,
   and that single Linear is bigger than attention and feed-forward combined.
2. **Q/K/V are projected twice.** `CrossAttentionBlock` applies `k_proj`,
   `v_proj`, `query_proj`, and then `nn.MultiheadAttention` applies its own
   `in_proj` to the result. Two linear maps in a row collapse into one, so
   these 5 × 12,480 = 62,400 parameters (14%) do not add expressiveness. They
   only change how the optimization behaves. Dropping them would take the
   model to 376,578.
3. **`context_mlp` is dead weight at `context_traj = 0`.** It gets an empty
   slice, so its 12,544 parameters never receive a gradient.
4. **The image path does not grow with N or the latent grid.** Going from
   `image_tokens` 9 to 64 did not add parameters: the queries are a fixed
   sin-cos table. Only the compute grows (the key sequence is now
   2·64 + T instead of 2·9 + T).
5. **Almost everything scales with D².** Doubling `hidden_dim` to 128 gives
   about 4× the parameters, roughly 1.7M.
6. **Each extra TemporalBlock costs 78,976.** With the 2 blocks that the
   report describes the total would be 281,026. The report (Section 5.5 and
   Table 6) still says 2 blocks and `image_tokens = 9`, while the code has 4
   and the config has 64.

## Update (same day): shared AdaLN, no double projection

Points 1 and 2 above are now implemented in `networks/blocks.py`:

- `CrossAttentionBlock` has no `k_proj`/`v_proj`/`query_proj` any more. The
  normalized tokens go straight into `nn.MultiheadAttention`.
- The block no longer owns a `Linear(D, 8D)`. It takes a precomputed (B, 8D)
  modulation and adds its own learned (8D,) offset (zero-init). The
  modulation comes from `AdaLNSingle` (SiLU + bias-free Linear, zero-init),
  one per temporal stack, owned by the caller.
- `SpatialBlock` keeps a private `AdaLNSingle`, so its AdaLN is unchanged
  (weight in the Linear, bias in the offset, same 33,280 parameters).

New count with the same config: **278,274** (was 438,978).

| Module | Params |
|---|---:|
| `temporal_stack` (4 × (16,640 MHA + 16,576 FF + 512 offset)) | 134,912 |
| `adaLN` (shared, Linear(64 → 512) without bias) | 32,768 |
| `image_aggregator` (2,112 + SpatialBlock 33,728 + its AdaLN 32,768) | 68,608 |
| `context_mlp` + `pred_mlp` | 25,088 |
| `fm_time_mlp` + `frame_mlp` | 16,640 |
| output head | 258 |

Checkpoints from before this change do not load into the new model.

Later the same day the output head was replaced by DiT's `FinalLayer`
(`networks/DiT/dit.py`): LayerNorm without affine, shift/scale from
SiLU + Linear(64 → 128) of the diffusion-time embedding, then Linear(64 → 2),
all zero-initialized. The head goes from 258 to 8,450 parameters, so the
total is **286,466**.

## Checking it

With the project interpreter (`PYTHONPATH=.`, `LD_LIBRARY_PATH` set to the env's `lib`):

```python
from exp_navsim.denoiser import TrajectoryDenoiser
m = TrajectoryDenoiser(hidden_dim=64, image_encoder_channels=16,
                       context_traj=0, context_images=2, image_tokens=64)
print(sum(p.numel() for p in m.parameters()))            # 286466 (438978 before the update)
for name, sub in m.named_children():
    print(f"{name:20s} {sum(p.numel() for p in sub.parameters()):>8,}")
```
