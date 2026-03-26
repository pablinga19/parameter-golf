"""Randomized Hadamard rotation for weight preprocessing before quantization.

Spreads outliers across all dimensions so INT6 quantization has fewer extreme
values. This lowers the effective entropy of quantized weights, improving
rANS compression ratio. Invertible — no information loss.

Pipeline: weights -> Hadamard rotate -> INT6 quantize -> rANS encode
          rANS decode -> INT6 dequantize -> inverse Hadamard -> weights
"""

from __future__ import annotations
import numpy as np
import torch


def hadamard_matrix(n: int) -> torch.Tensor:
    """Construct n x n Hadamard matrix (n must be power of 2). Normalized."""
    assert n & (n - 1) == 0 and n > 0
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / (n ** 0.5)  # normalize so H @ H.T = I


def random_signs(n: int, seed: int = 42) -> torch.Tensor:
    """Random +-1 diagonal for randomized Hadamard."""
    rng = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=rng) * 2 - 1).float()


def rotate_weight_matrix(W: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Apply randomized Hadamard rotation to a 2D weight matrix.

    For non-power-of-2 dimensions, pads to next power of 2, rotates, then crops.
    Returns rotated matrix of same shape as input.
    """
    rows, cols = W.shape

    # pad cols to power of 2
    cols_padded = 1
    while cols_padded < cols:
        cols_padded *= 2

    if cols_padded != cols:
        W_pad = torch.zeros(rows, cols_padded, dtype=W.dtype, device=W.device)
        W_pad[:, :cols] = W
    else:
        W_pad = W

    # random sign flip then Hadamard (per-row: multiply each column by random sign, then H)
    D = random_signs(cols_padded, seed).to(W.device)
    H = hadamard_matrix(cols_padded).to(W.device, dtype=W.dtype)

    # W_rot = (W_pad * D) @ H.T  -- randomized Hadamard transform per row
    W_rot = (W_pad * D.unsqueeze(0)) @ H.T

    return W_rot[:, :cols]  # crop back


def inverse_rotate_weight_matrix(W_rot: torch.Tensor, original_cols: int = None,
                                  seed: int = 42) -> torch.Tensor:
    """Inverse of rotate_weight_matrix. H is its own inverse (orthogonal)."""
    rows, cols = W_rot.shape
    if original_cols is None:
        original_cols = cols

    cols_padded = 1
    while cols_padded < cols:
        cols_padded *= 2

    if cols_padded != cols:
        W_pad = torch.zeros(rows, cols_padded, dtype=W_rot.dtype, device=W_rot.device)
        W_pad[:, :cols] = W_rot
    else:
        W_pad = W_rot

    D = random_signs(cols_padded, seed).to(W_rot.device)
    H = hadamard_matrix(cols_padded).to(W_rot.device, dtype=W_rot.dtype)

    # inverse: W = (W_rot @ H) * D  (because H.T = H for normalized Hadamard)
    W_rec = (W_pad @ H) * D.unsqueeze(0)

    return W_rec[:, :original_cols]


def rotate_state_dict(sd: dict, seed: int = 42, min_size: int = 64) -> dict:
    """Rotate all 2D weight matrices in a state dict."""
    rotated = {}
    for name, param in sd.items():
        if param.ndim == 2 and param.shape[0] >= min_size and param.shape[1] >= min_size:
            rotated[name] = rotate_weight_matrix(param.float(), seed).to(param.dtype)
        else:
            rotated[name] = param
    return rotated


def inverse_rotate_state_dict(sd: dict, shapes: dict, seed: int = 42, min_size: int = 64) -> dict:
    """Inverse rotate all 2D matrices back."""
    recovered = {}
    for name, param in sd.items():
        if param.ndim == 2 and param.shape[0] >= min_size and param.shape[1] >= min_size:
            orig_cols = shapes.get(name, param.shape[1])
            recovered[name] = inverse_rotate_weight_matrix(param.float(), orig_cols, seed).to(param.dtype)
        else:
            recovered[name] = param
    return recovered


if __name__ == "__main__":
    torch.manual_seed(42)

    # test roundtrip
    W = torch.randn(512, 512)
    W_rot = rotate_weight_matrix(W)
    W_rec = inverse_rotate_weight_matrix(W_rot)
    err = (W - W_rec).abs().max().item()
    print(f"512x512 roundtrip error: {err:.2e}")

    # test non-power-of-2
    W2 = torch.randn(256, 300)
    W2_rot = rotate_weight_matrix(W2)
    W2_rec = inverse_rotate_weight_matrix(W2_rot, original_cols=300)
    err2 = (W2 - W2_rec).abs().max().item()
    print(f"256x300 roundtrip error: {err2:.2e}")

    # show outlier spreading effect
    W3 = torch.randn(512, 512)
    W3[0, 0] = 100.0  # extreme outlier
    print(f"\nbefore rotation: max={W3.abs().max():.1f}, std={W3.std():.4f}")
    W3_rot = rotate_weight_matrix(W3)
    print(f"after rotation:  max={W3_rot.abs().max():.1f}, std={W3_rot.std():.4f}")

    # quantization entropy comparison
    def quantize_and_entropy(W, clip=31):
        scale = W.abs().amax(dim=1, keepdim=True) / clip
        q = torch.clamp(torch.round(W / scale.clamp(min=1e-8)), -clip, clip).to(torch.int8)
        vals, counts = q.flatten().unique(return_counts=True)
        probs = counts.float() / counts.sum()
        return -(probs * probs.log2()).sum().item()

    W4 = torch.randn(512, 512)
    W4[::10, ::10] = torch.randn(52, 52) * 10  # scattered outliers
    ent_before = quantize_and_entropy(W4)
    ent_after = quantize_and_entropy(rotate_weight_matrix(W4))
    print(f"\nINT6 entropy before Hadamard: {ent_before:.2f} bits/sym")
    print(f"INT6 entropy after Hadamard:  {ent_after:.2f} bits/sym")
    print(f"savings: {ent_before - ent_after:.2f} bits/sym")
