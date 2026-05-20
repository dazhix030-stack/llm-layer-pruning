# Experiment Summary

Residual injection for layer offloading — full experiment log.

## 1. Basic Results

**Goal:** Reduce VRAM while maintaining accuracy via NF4 quantization + layer offloading + residual injection.

| Model | Size | Layers Removed | Removal Rate | PPL | PPL Change | VRAM | Output Quality | Speed | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B Baseline | 0.5B | 0 | - | 2.97 | - | 1.88GB | ✓ Normal | 14 tok/s | float32 |
| 0.5B Even layers removed (residual injection) | 0.5B | 12/28 | 43% | 2.97 | +0.00 | 1.88GB | ✓ Identical | 14 tok/s | Full recovery with pre-injection |
| 0.5B Odd layers removed (residual injection) | 0.5B | 11/28 | 43% | 2.97 | +0.00 | 1.20GB | ✓ Identical | 14 tok/s | float32 |
| 3B Baseline | 3B | 0 | - | 3.04 | - | 5.85GB | ✓ Normal | - | float16 |
| 3B Even layers removed (residual injection) | 3B | 16/36 | 44% | 3.04 | +0.00 | 1.95GB | ✓ Identical | - | float16 |
| 3B NF4 Baseline | 3B | 0 | - | 3.46 | - | 1.92GB | ✓ Normal | 13.6 tok/s | NF4 |
| 3B NF4 + Layer offload + Residual injection (no KV cache) | 3B | 16/36 | 44% | 3.43 | -0.03 | 1.92GB | ✓ Identical | 6 tok/s | No KV cache |
| 3B NF4 + Disk offload (no KV cache) | 3B | 16/36 | 44% | 3.43 | -0.03 | 1.34GB | ✓ Identical | 1.5 tok/s | Disk I/O |
| **3B NF4 + Disk offload + KV cache** | **3B** | **16/36** | **44%** | **3.43** | **-0.03** | **standby 1.34GB / inference 1.93GB** | **✓ Identical** | **16.6 tok/s** | **kv_residual_model.py** |
| 7B NF4 Baseline | 7B | 0 | - | 3.56 | - | 5.20GB | ✓ Normal | 3 tok/s | NF4 |
| 7B NF4 Even layers removed (residual injection) | 7B | 12/28 | 43% | 3.57 | +0.01 | 3.85GB | ✓ Identical | - | NF4 |

## 2. Key Findings

| # | Finding | Result | Notes |
|---|---|---|---|
| 1 | Injecting residual **after** the layer | ❌ | Order matters |
| 1 | Injecting residual **before** the layer | ✓ | 1-line change, dramatic improvement |
| 2 | W_Q and W_gate spatial alignment | ❌ | τ and gate need separate handling |
| 3 | Residual injection works after NF4 quantization | ✓ | Works with NF4 |
| 4 | NF4 layer RAM→VRAM transfer error | ❌ | bitsandbytes constraint |
| 5 | Disk→VRAM direct load is possible | ✓ | state_dict route fails |
| 6 | Bottleneck is compute, not disk I/O | ⚠ | Caused by missing KV cache |
| 7 | Async prefetch works | ✓ | Load is not the bottleneck |

## 3. Speed Comparison

| Method | VRAM | Speed | PPL | Accuracy | Usability | Notes |
|---|---|---|---|---|---|---|
| HuggingFace NF4 Baseline | 1.92GB | 13.6 tok/s | 3.46 | ✓ | ◎ | Baseline |
| HuggingFace custom generate (no KV cache) | 1.92GB | 0.3 tok/s | 3.46 | ✓ | ✗ | No KV cache |
| HuggingFace model.generate | 1.92GB | 14 tok/s | 3.46 | ✓ | ◎ | With KV cache |
| Residual injection + GenerationMixin (no KV cache) | 1.93GB | 6 tok/s | 3.43 | ✓ | △ | No VRAM reduction |
| Residual injection + Disk offload (no KV cache) | 1.34GB | 1.5 tok/s | 3.43 | ✓ | △ | VRAM -30% |
| **Residual injection + Disk offload + KV cache** | **standby 1.34GB** | **16.6 tok/s** | **3.43** | **✓** | **◎** | **VRAM -31%, speed +22%** |
| llama.cpp full GPU | 2.62GB | 68 tok/s | - | ✓ | ◎ | KV cache optimized |
| llama.cpp layer removal GGUF (no residual) | 1.91GB | 108 tok/s | - | ✗ | ✗ | Output collapse |
| Layer removal only (no residual) HuggingFace | 1.94GB | 6.5 tok/s | 425 | ✗ | ✗ | Accuracy collapse |

## 4. Open Problems

| # | Problem | Cause | Candidate Solutions |
|---|---|---|---|
| 1 | VRAM reduction + speed + accuracy simultaneously | Removed layers still need to run for residual computation → compute unchanged | LoRA correction / constant residual approximation |
| 2 | Constant residual approximation failed | Layer 2 norm=3317, scale varies greatly per layer | Per-layer normalization and retry |
| 3 | Residual injection in llama.cpp | Cannot extract per-layer residuals from Python API | LoRA adapter format |
