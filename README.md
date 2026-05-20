# LLM Layer Offload with Residual Injection

Training-free VRAM reduction for local LLMs via disk offloading + residual correction.

## What it does

Offloads every other transformer layer to disk. At inference time, instead of running those layers normally, it computes only their residual contribution `F(x) = layer(x) - x` and injects it into the next active layer. No training required.

```
Normal:   x → Layer_n → Layer_n+1 → ...
Ours:     x → [Layer_n skipped]
                    ↓ load from disk, compute residual only
              Layer_n+1(x + α · F_n(x_before)) → ...
```

## Results (Qwen2.5-3B, NF4 quantization, RTX 4050 6GB)

| Method | Standby VRAM | Inference VRAM | PPL | Speed |
|---|---|---|---|---|
| Baseline (NF4) | 1.92 GB | 1.92 GB | 3.46 | 13.6 tok/s |
| **Residual Injection + KV Cache** | **1.34 GB** | 1.93 GB | **3.43** | **16.6 tok/s** |
| Disk Offload (no KV cache) | 1.34 GB | 1.34 GB | 3.43 | 1.5 tok/s |

- Standby VRAM: **-31%**
- Inference speed: **+22%** (with KV cache variant)
- PPL: no degradation (slightly improved)
- No training, no calibration data

## Files

| File | Description |
|---|---|
| `kv_residual_model.py` | Main implementation with KV cache support |
| `direct_vram_model.py` | Earlier version, async prefetch, no KV cache |

## Usage

```bash
python kv_residual_model.py --model /path/to/Qwen2.5-3B
```

## Requirements

- Python 3.12
- PyTorch 2.x
- transformers
- bitsandbytes
- GPU with CUDA (tested on RTX 4050 6GB)

## Limitations

- Tested on short sequences (~100 tokens). Long-context behavior is untested.
- Residual injection adds overhead; standby VRAM savings disappear during inference (weights are reloaded).
- Speed improvement comes from reduced memory pressure, not fewer computations.

## Notes

The key insight: in residual networks, each layer's contribution `F(x) = layer(x) - x` can be computed independently and injected into a later layer without retraining. Layers 2, 4, 6, ... 32 (16 of 36 total) are offloaded to disk. Their residuals are computed on-demand and added to the input of the next active layer.

Closest prior work: [KV-Direct (arXiv:2603.19664)](https://arxiv.org/abs/2603.19664) replaces KV cache with residual checkpoints. This repo targets model weights instead.
