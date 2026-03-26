# PPM-D Integration Guide

Drop-in replacement for the n-gram backoff section in PR #761's train_gpt.py.

## What changes

Replace lines ~1148-1198 (hash computation + backoff + linear mixing + table update)
with PPM-D Numba predict + logistic mixing + PPM-D Numba update.

## Setup

Add to imports at top of train_gpt.py:
```python
from ppmd_numba import PPMDNumba
from mixer import logistic_mix, entropy_adaptive_alpha
```

## Init (replace lines ~1080-1094)

Replace the ctx_tables/full_tables init with:
```python
if use_ngram:
    val_np = val_tokens.cpu().numpy()
    ppmd = PPMDNumba(
        max_order=ngram_order,
        min_order=ngram_min_order,
        num_buckets=ngram_buckets,
        min_count=ngram_min_count,
    )
    # warmup numba JIT (bounded by actual data length)
    _wend = min(len(val_np), 100)
    _wj = np.arange(min(10, _wend), min(len(val_np), 1000), dtype=np.int64)
    if _wend > 0 and len(_wj) > 0:
        ppmd.update_tables(val_np, 0, _wend)
        ppmd.predict_blended(val_np, _wj, len(_wj))
    # reset tables after warmup
    ppmd = PPMDNumba(
        max_order=ngram_order,
        min_order=ngram_min_order,
        num_buckets=ngram_buckets,
        min_count=ngram_min_count,
    )
```

## Predict + Mix (replace lines ~1148-1190)

```python
# PPM-D blended prediction (replaces pure backoff)
p_blend, has_match = ppmd.predict_blended(val_np, global_j, n_seg)

if has_match.any():
    if ngram_entropy:
        alpha = np.clip(
            alpha_per_tok[has_match], 0.0, 0.95)
    else:
        alpha = ngram_alpha
    # logistic mixing (replaces linear)
    seg_model_p[has_match] = logistic_mix(
        seg_model_p[has_match], p_blend[has_match], alpha)

seg_nll_np = -np.log(np.clip(seg_model_p, 1e-12, 1.0))
```

## Table update (replace lines ~1192-1198)

```python
ppmd.update_tables(val_np, int(global_j[0]), int(global_j[-1]) + 1)
```

## Env vars

Same as before, plus:
- MIXER_MODE=ppmd (default) or MIXER_MODE=linear (for A/B comparison)

## Expected results

PPM-D blending validated at -0.51 BPB (-7%) over pure backoff on 100K tokens.
Numba: 460K tok/s update, 1.15M tok/s predict. 62M tokens in ~200s.
