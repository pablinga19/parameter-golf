import numpy as np


def layer_bit_assignment(num_layers, early_bits=4, late_bits=6,
                         transition_layer=None):
    """assign per-layer bit widths. early layers get fewer bits (local stats,
    robust to noise), late layers get more (semantic structure, fragile).

    returns array of bit widths per layer.
    """
    if transition_layer is None:
        transition_layer = num_layers // 3
    assert 0 <= transition_layer <= num_layers
    assert early_bits >= 2 and late_bits >= 2

    bits = np.zeros(num_layers, dtype=np.int32)
    bits[:transition_layer] = early_bits
    bits[transition_layer:] = late_bits
    return bits


def estimate_savings(num_layers, params_per_layer, early_bits=4, late_bits=6,
                     transition_layer=None):
    bits = layer_bit_assignment(num_layers, early_bits, late_bits, transition_layer)
    uniform_bytes = num_layers * params_per_layer * 6 / 8
    hetero_bytes = sum(params_per_layer * b / 8 for b in bits)
    saved = uniform_bytes - hetero_bytes
    return {
        'bits_per_layer': bits.tolist(),
        'uniform_bytes': int(uniform_bytes),
        'hetero_bytes': int(hetero_bytes),
        'saved_bytes': int(saved),
        'saved_pct': saved / uniform_bytes * 100,
    }


def quantize_per_row(tensor, clip_range):
    """per-row symmetric quantization at given clip range."""
    row_max = np.abs(tensor).max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, 1e-8)
    scale = row_max / clip_range
    q = np.clip(np.round(tensor / scale), -clip_range, clip_range)
    return q, scale


def quantize_hetero(state_dict_arrays, layer_names, bits_per_layer):
    """quantize a dict of weight arrays with per-layer bit widths.

    layer_names: list mapping each key to its layer index.
    returns quantized arrays + scales + bit assignments.
    """
    result = {}
    for name, arr in state_dict_arrays.items():
        layer_idx = layer_names.get(name, -1)
        if layer_idx < 0 or arr.ndim != 2:
            result[name] = {'array': arr, 'bits': 0, 'scale': None}
            continue

        bits = bits_per_layer[layer_idx]
        clip = (1 << (bits - 1)) - 1  # INT4 -> 7, INT6 -> 31
        q, scale = quantize_per_row(arr, clip)
        mse = ((arr - q * scale) ** 2).mean()
        result[name] = {'array': q, 'bits': bits, 'scale': scale, 'mse': float(mse)}

    return result


if __name__ == "__main__":
    r = estimate_savings(11, 262144, early_bits=4, late_bits=6)
    print(f"11-layer model, 262K params/layer:")
    print(f"  bits: {r['bits_per_layer']}")
    print(f"  uniform INT6: {r['uniform_bytes']:,} bytes ({r['uniform_bytes']/1e6:.1f} MB)")
    print(f"  hetero INT4/6: {r['hetero_bytes']:,} bytes ({r['hetero_bytes']/1e6:.1f} MB)")
    print(f"  saved: {r['saved_bytes']:,} bytes ({r['saved_pct']:.1f}%)")

    # MSE comparison on random weights
    np.random.seed(42)
    w = np.random.randn(512, 512).astype(np.float32) * 0.02
    q4, s4 = quantize_per_row(w, 7)   # INT4
    q6, s6 = quantize_per_row(w, 31)  # INT6
    mse4 = ((w - q4 * s4) ** 2).mean()
    mse6 = ((w - q6 * s6) ** 2).mean()
    print(f"\n512x512 gaussian weights:")
    print(f"  INT4 MSE: {mse4:.2e}")
    print(f"  INT6 MSE: {mse6:.2e}")
    print(f"  ratio: {mse4/mse6:.1f}x worse")
