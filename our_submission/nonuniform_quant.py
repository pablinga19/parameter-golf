import numpy as np


def build_lloyd_max_codebook(weights_flat, n_levels=64, max_iter=50):
    weights = weights_flat.astype(np.float64)
    if weights.size == 0:
        raise ValueError("empty weights")
    if not np.isfinite(weights).all():
        raise ValueError("non-finite weights")

    wmin, wmax = weights.min(), weights.max()
    codebook = np.linspace(wmin, wmax, n_levels)
    prev_mse = float('inf')

    for _ in range(max_iter):
        diffs = np.abs(weights[:, None] - codebook[None, :])
        assignments = diffs.argmin(axis=1)
        new_codebook = np.zeros(n_levels)
        for i in range(n_levels):
            mask = assignments == i
            if mask.any():
                new_codebook[i] = weights[mask].mean()
            else:
                new_codebook[i] = codebook[i]
        mse = ((weights - new_codebook[assignments]) ** 2).mean()
        if abs(prev_mse - mse) < 1e-12:
            break
        prev_mse = mse
        codebook = np.sort(new_codebook)

    return codebook


def quantize_with_codebook(weights, codebook, per_row=True):
    if per_row:
        assert weights.ndim == 2
        row_max = np.abs(weights).max(axis=1, keepdims=True)
        row_max = np.maximum(row_max, 1e-8)
        scales = row_max.ravel().astype(np.float32)
        normalized = weights / row_max
    else:
        scale = max(np.abs(weights).max(), 1e-8)
        scales = np.array([scale], dtype=np.float32)
        normalized = weights / scale

    flat = normalized.ravel()
    diffs = np.abs(flat[:, None] - codebook[None, :])
    indices = diffs.argmin(axis=1).astype(np.uint8)
    return indices.reshape(weights.shape), scales, codebook


def dequantize_with_codebook(indices, scales, codebook, shape, per_row=True):
    values = codebook[indices.ravel()].reshape(shape)
    if per_row:
        assert len(scales) == shape[0]
        values = values * scales[:, None]
    else:
        values = values * scales[0]
    return values.astype(np.float32)
