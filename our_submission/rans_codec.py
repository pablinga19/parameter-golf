"""rANS entropy coder for INT6 weights. Replaces zstd/LZMA after quantization."""

from __future__ import annotations
import numpy as np

# rANS parameters: 32-bit state, 16-bit precision, byte renormalization
PROB_BITS = 16
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 23


def build_freq_table(data: np.ndarray, num_symbols: int = 63):
    """Build frequency table from INT6 data (-31..+31 mapped to 0..62)."""
    assert data.min() >= -31 and data.max() <= 31, f"values out of INT6 range: [{data.min()}, {data.max()}]"
    shifted = (data.astype(np.int32) + 31).ravel()
    counts = np.bincount(shifted, minlength=num_symbols).astype(np.int64)
    counts = np.maximum(counts, 1)  # no zeros

    # scale to PROB_SCALE
    total = counts.sum()
    freqs = np.zeros(num_symbols, dtype=np.int64)
    for i in range(num_symbols):
        freqs[i] = max(1, int(counts[i] * PROB_SCALE / total))

    # fix rounding
    while freqs.sum() > PROB_SCALE:
        freqs[np.argmax(freqs)] -= 1
    while freqs.sum() < PROB_SCALE:
        freqs[np.argmax(freqs)] += 1

    cumfreqs = np.zeros(num_symbols + 1, dtype=np.int64)
    cumfreqs[1:] = np.cumsum(freqs)
    return freqs, cumfreqs


def rans_enc_put(state: int, sym: int, freqs, cumfreqs, outbuf: bytearray) -> int:
    """Encode one symbol. Returns new state."""
    freq = int(freqs[sym])
    cumf = int(cumfreqs[sym])

    # renormalize: emit bytes so state stays in valid range after encoding
    # threshold = ((RANS_L >> PROB_BITS) << 8) * freq = freq << 15
    x_max = freq << 15
    while state >= x_max:
        outbuf.append(state & 0xFF)
        state >>= 8

    # encode: C(s,x) = (x // freq) << PROB_BITS + (x % freq) + cumf
    return ((state // freq) << PROB_BITS) + (state % freq) + cumf


def rans_dec_get(state: int, freqs, cumfreqs, cum2sym, inbuf, pos: int):
    """Decode one symbol. Returns (symbol, new_state, new_pos)."""
    # find symbol from slot
    slot = state & (PROB_SCALE - 1)
    sym = int(cum2sym[slot])
    freq = int(freqs[sym])
    cumf = int(cumfreqs[sym])

    # decode: D(x) = freq * (x >> PROB_BITS) + (x & mask) - cumf
    state = freq * (state >> PROB_BITS) + slot - cumf

    # renormalize
    while state < RANS_L and pos < len(inbuf):
        state = (state << 8) | inbuf[pos]
        pos += 1

    return sym, state, pos


def rans_encode(data: np.ndarray, freqs: np.ndarray, cumfreqs: np.ndarray) -> bytes:
    """Encode INT6 array. Returns compressed bytes."""
    shifted = (data.astype(np.int32) + 31).ravel()
    n = len(shifted)
    outbuf = bytearray()

    state = RANS_L  # initial state

    # encode in reverse
    for i in range(n - 1, -1, -1):
        state = rans_enc_put(state, int(shifted[i]), freqs, cumfreqs, outbuf)

    # flush state (4 bytes, big-endian)
    for _ in range(4):
        outbuf.append(state & 0xFF)
        state >>= 8

    outbuf.reverse()
    return bytes(outbuf)


def rans_decode(compressed: bytes, n: int, freqs: np.ndarray,
                cumfreqs: np.ndarray, num_symbols: int = 63) -> np.ndarray:
    """Decode rANS bytes back to INT6 array."""
    cum2sym = np.zeros(PROB_SCALE, dtype=np.uint8)
    for s in range(num_symbols):
        cum2sym[cumfreqs[s]:cumfreqs[s + 1]] = s

    buf = bytearray(compressed)
    if len(buf) < 4:
        raise ValueError(f"compressed stream too short: {len(buf)} bytes")
    pos = 0

    state = 0
    for _ in range(4):
        state = (state << 8) | buf[pos]
        pos += 1

    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        sym, state, pos = rans_dec_get(state, freqs, cumfreqs, cum2sym, buf, pos)
        out[i] = sym - 31

    if state < RANS_L and pos < len(buf):
        raise ValueError("rANS decode ended with unconsumed bytes")

    return out


def compress_weights(weights_int6: np.ndarray):
    """Compress INT6 tensor. Returns (bytes, freqs, cumfreqs, shape)."""
    shape = weights_int6.shape
    freqs, cumfreqs = build_freq_table(weights_int6.ravel())
    comp = rans_encode(weights_int6, freqs, cumfreqs)
    return comp, freqs, cumfreqs, shape


def decompress_weights(comp: bytes, freqs, cumfreqs, shape):
    """Decompress back to INT6 tensor."""
    n = 1
    for s in shape:
        n *= s
    return rans_decode(comp, n, freqs, cumfreqs).reshape(shape)


if __name__ == "__main__":
    import time
    np.random.seed(42)

    # small roundtrip test first
    w_small = np.array([0, 1, -1, 5, -5, 31, -31, 0, 0, 0, 3, -3, 0, 1, -1, 0, 2, -2, 0, 0], dtype=np.int8)
    c, f, cf, s = compress_weights(w_small)
    r = decompress_weights(c, f, cf, s)
    assert np.array_equal(w_small, r), f"SMALL TEST FAILED\norig: {w_small}\nrec:  {r}"
    print("small roundtrip: PASS")

    # medium test with realistic distribution
    raw = np.random.standard_t(df=4, size=100_000)
    raw = raw / np.abs(raw).max() * 31
    w_med = np.clip(np.round(raw), -31, 31).astype(np.int8)
    c, f, cf, s = compress_weights(w_med)
    r = decompress_weights(c, f, cf, s)
    assert np.array_equal(w_med, r), "MEDIUM TEST FAILED"
    print("medium roundtrip (100K): PASS")

    # entropy stats
    vals, counts = np.unique(w_med, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    print(f"entropy: {entropy:.2f} bits/sym")
    print(f"compressed: {len(c):,} bytes ({len(c)*8/w_med.size:.2f} bits/sym)")
    print(f"ratio: {len(c)/w_med.nbytes:.3f}")

    # full-scale benchmark
    print("\nfull-scale benchmark (27M weights)...")
    raw = np.random.standard_t(df=4, size=27_000_000)
    raw = raw / np.abs(raw).max() * 31
    w_full = np.clip(np.round(raw), -31, 31).astype(np.int8)

    vals, counts = np.unique(w_full, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    print(f"entropy: {entropy:.2f} bits/sym")

    t0 = time.time()
    c, f, cf, s = compress_weights(w_full)
    t_enc = time.time() - t0
    print(f"encode: {t_enc:.1f}s, {len(c)/1e6:.1f} MB ({len(c)*8/w_full.size:.2f} bits/sym)")

    t0 = time.time()
    r = decompress_weights(c, f, cf, s)
    t_dec = time.time() - t0
    assert np.array_equal(w_full, r), "FULL TEST FAILED"
    print(f"decode: {t_dec:.1f}s, PERFECT roundtrip")
