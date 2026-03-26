import numpy as np

PROB_BITS = 16
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 23


def build_freq_table(data, num_symbols=63):
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


def rans_enc_put(state, sym, freqs, cumfreqs, outbuf):
    freq = int(freqs[sym])
    cumf = int(cumfreqs[sym])
    x_max = freq << 15
    while state >= x_max:
        outbuf.append(state & 0xFF)
        state >>= 8
    return ((state // freq) << PROB_BITS) + (state % freq) + cumf


def rans_dec_get(state, freqs, cumfreqs, cum2sym, inbuf, pos):
    slot = state & (PROB_SCALE - 1)
    sym = int(cum2sym[slot])
    freq = int(freqs[sym])
    cumf = int(cumfreqs[sym])
    state = freq * (state >> PROB_BITS) + slot - cumf
    while state < RANS_L and pos < len(inbuf):
        state = (state << 8) | inbuf[pos]
        pos += 1
    return sym, state, pos


def rans_encode(data, freqs, cumfreqs):
    shifted = (data.astype(np.int32) + 31).ravel()
    n = len(shifted)
    outbuf = bytearray()
    state = RANS_L

    for i in range(n - 1, -1, -1):
        state = rans_enc_put(state, int(shifted[i]), freqs, cumfreqs, outbuf)

    for _ in range(4):
        outbuf.append(state & 0xFF)
        state >>= 8
    outbuf.reverse()
    return bytes(outbuf)


def rans_decode(compressed, n, freqs, cumfreqs, num_symbols=63):
    cum2sym = np.zeros(PROB_SCALE, dtype=np.uint8)
    for s in range(num_symbols):
        cum2sym[cumfreqs[s]:cumfreqs[s + 1]] = s

    buf = bytearray(compressed)
    if len(buf) < 4:
        raise ValueError(f"stream too short: {len(buf)}")
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
        raise ValueError("unconsumed bytes after decode")
    return out


def compress_weights(w):
    shape = w.shape
    freqs, cumfreqs = build_freq_table(w.ravel())
    comp = rans_encode(w, freqs, cumfreqs)
    return comp, freqs, cumfreqs, shape


def decompress_weights(comp, freqs, cumfreqs, shape):
    n = 1
    for s in shape:
        n *= s
    return rans_decode(comp, n, freqs, cumfreqs).reshape(shape)
