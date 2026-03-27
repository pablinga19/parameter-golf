import numpy as np
from numba import njit

PROB_BITS = 16
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 23


@njit(cache=True)
def _rans_encode(shifted, n, freqs, cumfreqs):
    # worst case: each symbol emits 4 bytes + 4 byte flush
    buf = np.zeros(n * 4 + 16, dtype=np.uint8)
    pos = 0
    state = RANS_L

    for i in range(n - 1, -1, -1):
        s = shifted[i]
        freq = freqs[s]
        cumf = cumfreqs[s]
        x_max = freq << 15
        while state >= x_max:
            buf[pos] = state & 0xFF
            pos += 1
            state >>= 8
        state = ((state // freq) << PROB_BITS) + (state % freq) + cumf

    for _ in range(4):
        buf[pos] = state & 0xFF
        pos += 1
        state >>= 8

    # reverse the output
    result = np.empty(pos, dtype=np.uint8)
    for i in range(pos):
        result[i] = buf[pos - 1 - i]
    return result


@njit(cache=True)
def _rans_decode(buf, n, freqs, cumfreqs, cum2sym):
    pos = 0
    state = 0
    for _ in range(4):
        state = (state << 8) | buf[pos]
        pos += 1

    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        slot = state & (PROB_SCALE - 1)
        s = cum2sym[slot]
        freq = freqs[s]
        cumf = cumfreqs[s]
        state = freq * (state >> PROB_BITS) + slot - cumf
        while state < RANS_L and pos < len(buf):
            state = (state << 8) | buf[pos]
            pos += 1
        out[i] = s - 31

    return out


def build_freq_table(data, num_symbols=63):
    if data.size == 0:
        raise ValueError("empty weight data")
    assert data.min() >= -31 and data.max() <= 31
    shifted = (data.astype(np.int32) + 31).ravel()
    counts = np.bincount(shifted, minlength=num_symbols).astype(np.int64)
    counts = np.maximum(counts, 1)

    total = counts.sum()
    freqs = np.zeros(num_symbols, dtype=np.int64)
    for i in range(num_symbols):
        freqs[i] = max(1, int(counts[i] * PROB_SCALE / total))

    while freqs.sum() > PROB_SCALE:
        freqs[np.argmax(freqs)] -= 1
    while freqs.sum() < PROB_SCALE:
        freqs[np.argmax(freqs)] += 1

    cumfreqs = np.zeros(num_symbols + 1, dtype=np.int64)
    cumfreqs[1:] = np.cumsum(freqs)
    return freqs, cumfreqs


def compress_weights(w):
    shape = w.shape
    freqs, cumfreqs = build_freq_table(w.ravel())
    shifted = (w.astype(np.int32) + 31).ravel()
    comp = _rans_encode(shifted, len(shifted), freqs, cumfreqs)
    return bytes(comp), freqs, cumfreqs, shape


def decompress_weights(comp, freqs, cumfreqs, shape):
    if len(comp) < 4:
        raise ValueError(f"stream too short: {len(comp)}")
    assert cumfreqs[0] == 0 and cumfreqs[-1] == PROB_SCALE
    n = 1
    for s in shape:
        n *= s
    buf = np.frombuffer(comp, dtype=np.uint8)

    cum2sym = np.zeros(PROB_SCALE, dtype=np.int64)
    for s in range(63):
        cum2sym[cumfreqs[s]:cumfreqs[s + 1]] = s

    return _rans_decode(buf, n, freqs, cumfreqs, cum2sym).reshape(shape)


if __name__ == "__main__":
    import time
    np.random.seed(42)

    # roundtrip test
    w = np.clip(np.round(np.random.randn(1000) * 10), -31, 31).astype(np.int8)
    c, f, cf, s = compress_weights(w)
    r = decompress_weights(c, f, cf, s)
    assert np.array_equal(w, r), "roundtrip FAILED"
    print("roundtrip: ok")

    # warmup
    compress_weights(w)
    decompress_weights(c, f, cf, s)

    # benchmark
    raw = np.random.standard_t(df=4, size=27_000_000)
    raw = raw / np.abs(raw).max() * 31
    w_full = np.clip(np.round(raw), -31, 31).astype(np.int8)

    t0 = time.time()
    c, f, cf, s = compress_weights(w_full)
    dt_enc = time.time() - t0

    t0 = time.time()
    r = decompress_weights(c, f, cf, s)
    dt_dec = time.time() - t0

    assert np.array_equal(w_full, r), "MISMATCH"
    print(f"encode: {dt_enc:.2f}s  decode: {dt_dec:.2f}s  size: {len(c)/1e6:.1f}MB  roundtrip: ok")
