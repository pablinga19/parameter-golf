"""
Numba-accelerated CTW-inspired mixer for Parameter Golf.

Replaces the pure-Python CTWMixer with flat numpy arrays and @njit kernels.
Target: process 62M tokens in under 10 minutes on a single CPU core.

Design (v2 — memory-efficient):
- Instead of per-token count arrays (num_buckets * vocab_size → 32GB!),
  use PPM-style dual hash tables:
    ctx_tables[depth][hash(context)] → total count at this context
    full_tables[depth][hash(context, token)] → count of this token at context
- KT estimator: P(token|ctx) = (count(ctx,tok) + 0.5) / (total(ctx) + V/2)
- Bottom-up CTW weighting: P_w = 0.5 * P_e + 0.5 * P_w(deeper)

Memory: 2 tables × 8 depths × 4M buckets × 4 bytes = 256 MB
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Core Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _hash_at_depth(tokens, pos, depth, primes, mask):
    """Hash `depth` tokens preceding position `pos`. Returns -1 if insufficient context."""
    if depth == 0:
        return np.int64(0)
    h = np.uint64(0)
    for k in range(depth):
        idx = pos - 1 - k
        if idx < 0:
            return np.int64(-1)
        h ^= primes[k] * np.uint64(tokens[idx])
    return np.int64(h & mask)


@njit(cache=True)
def _hash_full(ctx_hash_u64, token, prime, mask):
    """Hash context + target token."""
    return np.int64((ctx_hash_u64 ^ (prime * np.uint64(token))) & mask)


@njit(cache=True)
def _ctw_predict_update_batch(
    val_np,
    global_j,
    n_seg,
    max_depth,
    vocab_half,         # vocab_size * 0.5
    primes,
    mask,
    # Per-depth tables: ctx_tables[d], full_tables[d]
    ctx0, full0,
    ctx1, full1,
    ctx2, full2,
    ctx3, full3,
    ctx4, full4,
    ctx5, full5,
    ctx6, full6,
    ctx7, full7,
    do_update,          # bool: whether to update tables after predicting
):
    """Combined predict + optional update for a batch of tokens.

    Returns p_ctw array of shape (n_seg,).
    """
    ctx_arrays = (ctx0, ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7)
    full_arrays = (full0, full1, full2, full3, full4, full5, full6, full7)

    p_ctw = np.empty(n_seg, dtype=np.float64)
    vs_inv = 1.0 / (2.0 * vocab_half)  # 1/vocab_size

    for idx in range(n_seg):
        j = global_j[idx]
        if j <= 0 or j >= len(val_np):
            p_ctw[idx] = vs_inv
            continue

        next_token = val_np[j]

        # Collect KT predictions at each valid depth, bottom-up
        preds_n = 0
        # We'll store predictions in a small stack (max 8)
        p0 = 0.0; p1 = 0.0; p2 = 0.0; p3 = 0.0
        p4 = 0.0; p5 = 0.0; p6 = 0.0; p7 = 0.0
        preds = (p0, p1, p2, p3, p4, p5, p6, p7)

        # Also store keys for update
        ctx_keys_0 = np.int64(-1); ctx_keys_1 = np.int64(-1)
        ctx_keys_2 = np.int64(-1); ctx_keys_3 = np.int64(-1)
        ctx_keys_4 = np.int64(-1); ctx_keys_5 = np.int64(-1)
        ctx_keys_6 = np.int64(-1); ctx_keys_7 = np.int64(-1)
        full_keys_0 = np.int64(-1); full_keys_1 = np.int64(-1)
        full_keys_2 = np.int64(-1); full_keys_3 = np.int64(-1)
        full_keys_4 = np.int64(-1); full_keys_5 = np.int64(-1)
        full_keys_6 = np.int64(-1); full_keys_7 = np.int64(-1)

        # Depth 0: unigram (no context hash needed)
        ck = np.int64(0)
        prime_for_tok = primes[0]  # use first prime for token hash at depth 0
        fk = _hash_full(np.uint64(0), next_token, prime_for_tok, mask)
        c_ctx = np.float64(ctx_arrays[0][ck])
        c_full = np.float64(full_arrays[0][fk])
        p_kt = (c_full + 0.5) / (c_ctx + vocab_half)
        p0 = p_kt
        ctx_keys_0 = ck; full_keys_0 = fk
        preds_n = 1

        # Depths 1..max_depth
        for d in range(1, max_depth + 1):
            h = _hash_at_depth(val_np, j, d, primes, mask)
            if h < 0:
                break
            ck_d = h
            prime_idx = min(d, len(primes) - 1)
            fk_d = _hash_full(np.uint64(h), next_token, primes[prime_idx], mask)
            c_ctx_d = np.float64(ctx_arrays[d][ck_d])
            c_full_d = np.float64(full_arrays[d][fk_d])
            p_kt_d = (c_full_d + 0.5) / (c_ctx_d + vocab_half)

            if d == 1:
                p1 = p_kt_d; ctx_keys_1 = ck_d; full_keys_1 = fk_d
            elif d == 2:
                p2 = p_kt_d; ctx_keys_2 = ck_d; full_keys_2 = fk_d
            elif d == 3:
                p3 = p_kt_d; ctx_keys_3 = ck_d; full_keys_3 = fk_d
            elif d == 4:
                p4 = p_kt_d; ctx_keys_4 = ck_d; full_keys_4 = fk_d
            elif d == 5:
                p5 = p_kt_d; ctx_keys_5 = ck_d; full_keys_5 = fk_d
            elif d == 6:
                p6 = p_kt_d; ctx_keys_6 = ck_d; full_keys_6 = fk_d
            elif d == 7:
                p7 = p_kt_d; ctx_keys_7 = ck_d; full_keys_7 = fk_d
            preds_n = d + 1

        # Bottom-up CTW: P_w = 0.5 * P_e(d) + 0.5 * P_w(d+1)
        if preds_n == 0:
            p_ctw[idx] = vs_inv
        else:
            # Get deepest prediction
            if preds_n >= 8:   p_w = p7
            elif preds_n >= 7: p_w = p6
            elif preds_n >= 6: p_w = p5
            elif preds_n >= 5: p_w = p4
            elif preds_n >= 4: p_w = p3
            elif preds_n >= 3: p_w = p2
            elif preds_n >= 2: p_w = p1
            else:              p_w = p0

            # Mix from second-deepest up
            all_p = (p0, p1, p2, p3, p4, p5, p6, p7)
            for d in range(preds_n - 2, -1, -1):
                p_w = 0.5 * all_p[d] + 0.5 * p_w

            p_ctw[idx] = p_w

        # Update tables if requested
        if do_update:
            all_ck = (ctx_keys_0, ctx_keys_1, ctx_keys_2, ctx_keys_3,
                      ctx_keys_4, ctx_keys_5, ctx_keys_6, ctx_keys_7)
            all_fk = (full_keys_0, full_keys_1, full_keys_2, full_keys_3,
                      full_keys_4, full_keys_5, full_keys_6, full_keys_7)
            for d in range(preds_n):
                ck_u = all_ck[d]
                fk_u = all_fk[d]
                if ck_u >= 0:
                    ctx_arrays[d][ck_u] += 1
                    full_arrays[d][fk_u] += 1

    return p_ctw


@njit(cache=True)
def _ctw_update_only(
    val_np,
    scored_start,
    scored_end,
    max_depth,
    primes,
    mask,
    ctx0, full0, ctx1, full1, ctx2, full2, ctx3, full3,
    ctx4, full4, ctx5, full5, ctx6, full6, ctx7, full7,
):
    """Update CTW tables for a range of tokens (no prediction)."""
    ctx_arrays = (ctx0, ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7)
    full_arrays = (full0, full1, full2, full3, full4, full5, full6, full7)

    for j in range(scored_start, scored_end):
        if j <= 0 or j >= len(val_np):
            continue
        next_token = val_np[j]

        # Depth 0
        ck = np.int64(0)
        fk = _hash_full(np.uint64(0), next_token, primes[0], mask)
        ctx_arrays[0][ck] += 1
        full_arrays[0][fk] += 1

        # Depths 1..max_depth
        for d in range(1, max_depth + 1):
            h = _hash_at_depth(val_np, j, d, primes, mask)
            if h < 0:
                break
            ck_d = h
            prime_idx = min(d, len(primes) - 1)
            fk_d = _hash_full(np.uint64(h), next_token, primes[prime_idx], mask)
            ctx_arrays[d][ck_d] += 1
            full_arrays[d][fk_d] += 1


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

class NumbaCTWMixer:
    """Numba-accelerated CTW-inspired mixer (v2 — memory-efficient).

    Uses PPM-style dual hash tables instead of full per-token count arrays.
    Memory: ~256 MB for 4M buckets × 8 depths (vs 32 GB in v1).
    """

    MAX_DEPTHS = 8

    def __init__(
        self,
        max_depth: int = 7,
        vocab_size: int = 1024,
        num_buckets: int = 4_194_304,
        primes: list[int] | None = None,
    ):
        if num_buckets & (num_buckets - 1) != 0:
            raise ValueError(f"num_buckets must be a power of 2, got {num_buckets}")
        if max_depth >= self.MAX_DEPTHS:
            raise ValueError(f"max_depth must be < {self.MAX_DEPTHS}, got {max_depth}")

        self.max_depth = max_depth
        self.vocab_size = vocab_size
        self.num_buckets = num_buckets
        self.mask = np.uint64(num_buckets - 1)
        self.vocab_half = np.float64(vocab_size * 0.5)

        if primes is None:
            primes = [36313, 27191, 51647, 81929, 131071, 175447, 209591]
        self.primes = np.array(primes[:max(max_depth, 1)], dtype=np.uint64)

        # Dual hash tables per depth (same as PPM approach)
        self.ctx_tables: list[np.ndarray] = []
        self.full_tables: list[np.ndarray] = []
        for _ in range(self.MAX_DEPTHS):
            self.ctx_tables.append(np.zeros(num_buckets, dtype=np.uint32))
            self.full_tables.append(np.zeros(num_buckets, dtype=np.uint32))

    def _arrays(self):
        return (
            self.ctx_tables[0], self.full_tables[0],
            self.ctx_tables[1], self.full_tables[1],
            self.ctx_tables[2], self.full_tables[2],
            self.ctx_tables[3], self.full_tables[3],
            self.ctx_tables[4], self.full_tables[4],
            self.ctx_tables[5], self.full_tables[5],
            self.ctx_tables[6], self.full_tables[6],
            self.ctx_tables[7], self.full_tables[7],
        )

    def predict_batch(
        self,
        val_np: np.ndarray,
        global_j: np.ndarray,
        n_seg: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch prediction (score-first: predict without updating)."""
        p_ctw = _ctw_predict_update_batch(
            val_np, global_j, n_seg,
            self.max_depth, self.vocab_half, self.primes, self.mask,
            *self._arrays(),
            False,  # do_update=False
        )
        has_match = np.ones(n_seg, dtype=np.bool_)
        return p_ctw, has_match

    def predict_and_update_batch(
        self,
        val_np: np.ndarray,
        global_j: np.ndarray,
        n_seg: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict then immediately update (for non-score-first protocols)."""
        p_ctw = _ctw_predict_update_batch(
            val_np, global_j, n_seg,
            self.max_depth, self.vocab_half, self.primes, self.mask,
            *self._arrays(),
            True,  # do_update=True
        )
        has_match = np.ones(n_seg, dtype=np.bool_)
        return p_ctw, has_match

    def update_batch(self, val_np: np.ndarray, scored_start: int, scored_end: int):
        """Update tables after scoring (score-first protocol)."""
        _ctw_update_only(
            val_np, scored_start, scored_end,
            self.max_depth, self.primes, self.mask,
            *self._arrays(),
        )

    def memory_mb(self) -> float:
        ctx_bytes = sum(a.nbytes for a in self.ctx_tables)
        full_bytes = sum(a.nbytes for a in self.full_tables)
        return (ctx_bytes + full_bytes) / (1024 * 1024)


def warmup_jit():
    """Trigger JIT compilation with tiny workload."""
    mixer = NumbaCTWMixer(max_depth=2, vocab_size=4, num_buckets=16)
    tokens = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    gj = np.array([2, 3, 4, 5], dtype=np.int64)
    mixer.predict_batch(tokens, gj, 4)
    mixer.update_batch(tokens, 2, 6)
    return mixer


if __name__ == "__main__":
    import time

    print("Warming up JIT...")
    warmup_jit()

    # Benchmark: 1M tokens
    n_tokens = 1_000_000
    vocab = 1024
    np.random.seed(42)
    tokens = np.random.randint(0, vocab, size=n_tokens, dtype=np.int64)

    mixer = NumbaCTWMixer(max_depth=7, vocab_size=vocab, num_buckets=4_194_304)
    print(f"Memory: {mixer.memory_mb():.1f} MB")

    stride = 64
    seq_len = 1024
    total_scored = 0
    t0 = time.perf_counter()

    for ws in range(0, n_tokens - seq_len, stride):
        wlen = min(seq_len, n_tokens - 1 - ws)
        s = 0 if ws == 0 else max(wlen - stride, 0)
        seg_len = wlen - s
        if seg_len <= 0:
            continue
        global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
        p_ctw, _ = mixer.predict_batch(tokens, global_j, seg_len)
        mixer.update_batch(tokens, ws + s + 1, ws + wlen + 1)
        total_scored += seg_len

    elapsed = time.perf_counter() - t0
    rate = total_scored / elapsed
    print(f"Scored {total_scored:,} tokens in {elapsed:.2f}s ({rate:,.0f} tok/s)")
    print(f"Extrapolated 62M tokens: {62_000_000 / rate:.1f}s ({62_000_000 / rate / 60:.1f} min)")
