"""
Numba-accelerated PPM-D mixer for Parameter Golf.

Replaces the Python-loop PPMDMixer with JIT-compiled functions.
Target: 500K+ tok/s for update, 200K+ tok/s for predict.
Current Python baseline: 3K tok/s update, 30K tok/s predict.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange, types
from numba.typed import List as NumbaList


# ---------------------------------------------------------------------------
# Hash functions (Numba-compatible)
# ---------------------------------------------------------------------------

PRIMES = np.array([36313, 27191, 51647, 81929, 131071, 175447, 209591], dtype=np.uint64)


@njit(cache=True)
def hash_context(val_np, pos, ctx_len, primes, mask):
    """Hash ctx_len tokens PRECEDING position pos."""
    h = np.uint64(0)
    for k in range(ctx_len):
        idx = pos - 1 - k
        if idx < 0:
            return np.uint64(0xFFFFFFFFFFFFFFFF)  # sentinel
        h ^= primes[k] * np.uint64(val_np[idx])
    return h & mask


@njit(cache=True)
def hash_full(ctx_hash, next_token, ctx_len, primes, mask):
    """Hash = context_hash XOR prime[ctx_len] * next_token."""
    if ctx_hash == np.uint64(0xFFFFFFFFFFFFFFFF):
        return ctx_hash
    prime_idx = min(ctx_len, len(primes) - 1)
    h = ctx_hash ^ (primes[prime_idx] * np.uint64(next_token))
    return h & mask


# ---------------------------------------------------------------------------
# Numba PPM-D update (the critical bottleneck: 3K → 500K+ tok/s)
# ---------------------------------------------------------------------------

@njit(cache=True)
def ppmd_update_batch(
    val_np,          # uint16 or int32 array of all tokens
    start,           # first token position to update
    end,             # last token position (exclusive)
    n_orders,        # number of orders
    min_order,       # minimum n-gram order
    primes,          # hash primes array
    mask,            # bucket mask (num_buckets - 1)
    ctx_tables,      # list of n_orders uint32 arrays
    full_tables,     # list of n_orders uint32 arrays
    unique_tables,   # list of n_orders uint16 arrays
):
    """Update all order tables for positions [start, end).

    Score-first protocol: this is called AFTER scoring the segment.
    """
    for j in range(start, end):
        next_token = np.uint64(val_np[j])
        for oi in range(n_orders):
            order = min_order + oi
            ctx_len = order - 1
            if j < ctx_len:
                continue

            # Hash preceding tokens
            ctx_key = np.uint64(0)
            for k in range(ctx_len):
                ctx_key ^= primes[k] * np.uint64(val_np[j - 1 - k])
            ctx_key &= mask

            # Full key = context + target
            prime_idx = min(ctx_len, len(primes) - 1)
            full_key = ctx_key ^ (primes[prime_idx] * next_token)
            full_key &= mask

            # Track unique next-tokens for PPM-D escape
            ct = ctx_tables[oi]
            ft = full_tables[oi]
            ut = unique_tables[oi]

            if ft[full_key] == 0:
                if ut[ctx_key] < 65535:
                    ut[ctx_key] += 1

            ct[ctx_key] += 1
            ft[full_key] += 1


# ---------------------------------------------------------------------------
# Numba PPM-D predict (30K → 200K+ tok/s)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def ppmd_predict_batch(
    val_np,          # all tokens
    global_j,        # int64 array: positions of tokens to predict
    n_seg,           # number of tokens
    n_orders,        # number of orders
    min_order,       # minimum n-gram order
    min_count,       # minimum context count for match
    primes,          # hash primes
    mask,            # bucket mask
    ctx_tables,      # list of uint32 arrays
    full_tables,     # list of uint32 arrays
    unique_tables,   # list of uint16 arrays
    depth_boost,     # float64 array of per-order boost weights
):
    """Predict using PPM-D blending across all orders.

    Returns:
        p_blend: float64 array of blended n-gram probabilities
        has_match: bool array
    """
    p_blend = np.zeros(n_seg, dtype=np.float64)
    has_match = np.zeros(n_seg, dtype=np.bool_)

    for idx in prange(n_seg):
        j = global_j[idx]
        next_token = np.uint64(val_np[j])

        total_weight = 0.0
        weighted_p = 0.0
        found_any = False

        # Iterate orders highest-first for backoff priority
        for oi in range(n_orders - 1, -1, -1):
            order = min_order + oi
            ctx_len = order - 1

            if j < ctx_len:
                continue

            # Hash preceding tokens
            ctx_key = np.uint64(0)
            for k in range(ctx_len):
                ctx_key ^= primes[k] * np.uint64(val_np[j - 1 - k])
            ctx_key &= mask

            ctx_count = float(ctx_tables[oi][ctx_key])
            if ctx_count < float(min_count):
                continue

            # Full key
            prime_idx = min(ctx_len, len(primes) - 1)
            full_key = ctx_key ^ (primes[prime_idx] * next_token)
            full_key &= mask

            full_count = float(full_tables[oi][full_key])
            unique_count = float(unique_tables[oi][ctx_key])

            # P(next | context) = count(context+next) / count(context)
            p = min(full_count, ctx_count) / max(ctx_count, 1.0)
            p = max(0.0, min(1.0, p))

            # PPM-D escape: confidence = 1 - (unique / (total + unique))
            escape = unique_count / (ctx_count + unique_count + 1e-10)
            confidence = (1.0 - escape) * depth_boost[oi]

            weighted_p += confidence * p
            total_weight += confidence
            found_any = True

        if found_any and total_weight > 1e-10:
            p_blend[idx] = weighted_p / total_weight
            has_match[idx] = True

    return p_blend, has_match


# ---------------------------------------------------------------------------
# Python wrapper class
# ---------------------------------------------------------------------------

class PPMDNumba:
    """Numba-accelerated PPM-D mixer.

    Drop-in replacement for PPMDMixer with identical interface.
    """

    def __init__(
        self,
        max_order: int = 7,
        min_order: int = 2,
        num_buckets: int = 4_194_304,
        min_count: int = 2,
        depth_boost_base: float = 2.0,
    ):
        if num_buckets & (num_buckets - 1) != 0:
            raise ValueError(f"num_buckets must be power of 2, got {num_buckets}")

        self.max_order = max_order
        self.min_order = min_order
        self.num_buckets = num_buckets
        self.min_count = min_count
        self.n_orders = max_order - min_order + 1
        self.mask = np.uint64(num_buckets - 1)
        self.primes = PRIMES[:max_order].copy()

        # Numba-typed lists for table arrays
        self.ctx_tables = NumbaList()
        self.full_tables = NumbaList()
        self.unique_tables = NumbaList()
        for _ in range(self.n_orders):
            self.ctx_tables.append(np.zeros(num_buckets, dtype=np.uint32))
            self.full_tables.append(np.zeros(num_buckets, dtype=np.uint32))
            self.unique_tables.append(np.zeros(num_buckets, dtype=np.uint16))

        # Per-order depth boost weights (higher orders get exponential boost)
        self.depth_boost = np.array([depth_boost_base ** oi for oi in range(self.n_orders)], dtype=np.float64)

    def predict_blended(
        self,
        val_np: np.ndarray,
        global_j: np.ndarray,
        n_seg: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict using Numba-accelerated PPM-D blending."""
        return ppmd_predict_batch(
            val_np, global_j, n_seg,
            self.n_orders, self.min_order, self.min_count,
            self.primes, self.mask,
            self.ctx_tables, self.full_tables, self.unique_tables,
            self.depth_boost,
        )

    def update_tables(
        self,
        val_np: np.ndarray,
        scored_start: int,
        scored_end: int,
    ):
        """Update tables using Numba-accelerated batch update."""
        ppmd_update_batch(
            val_np, scored_start, scored_end,
            self.n_orders, self.min_order,
            self.primes, self.mask,
            self.ctx_tables, self.full_tables, self.unique_tables,
        )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark():
    """Benchmark Numba PPM-D vs Python PPM-D."""
    np.random.seed(42)
    n_tokens = 500_000
    val_np = np.random.randint(0, 1024, size=n_tokens, dtype=np.int32)

    ppmd = PPMDNumba(max_order=7, min_order=2, num_buckets=4_194_304)

    # Warmup JIT
    print("Warming up Numba JIT...", flush=True)
    ppmd.update_tables(val_np, 0, 1000)
    global_j_warmup = np.arange(100, 1000, dtype=np.int64)
    ppmd.predict_blended(val_np, global_j_warmup, len(global_j_warmup))
    print("JIT warm.", flush=True)

    # Benchmark update
    ppmd2 = PPMDNumba(max_order=7, min_order=2, num_buckets=4_194_304)
    t0 = time.time()
    ppmd2.update_tables(val_np, 0, n_tokens)
    t_update = time.time() - t0
    update_rate = n_tokens / t_update

    # Benchmark predict
    global_j = np.arange(10, n_tokens, dtype=np.int64)
    t0 = time.time()
    p_blend, has_match = ppmd2.predict_blended(val_np, global_j, len(global_j))
    t_predict = time.time() - t0
    predict_rate = len(global_j) / t_predict

    print(f"\n{'='*60}")
    print(f"PPM-D Numba Benchmark ({n_tokens:,} tokens)")
    print(f"{'='*60}")
    print(f"Update:  {update_rate:,.0f} tok/s ({t_update:.3f}s)")
    print(f"Predict: {predict_rate:,.0f} tok/s ({t_predict:.3f}s)")
    print(f"Hit rate: {has_match.mean()*100:.1f}%")
    print(f"\nProjected 62M token times:")
    print(f"  Update:  {62_000_000 / update_rate:.1f}s")
    print(f"  Predict: {62_000_000 / predict_rate:.1f}s")
    print(f"  Total:   {62_000_000 / update_rate + 62_000_000 / predict_rate:.1f}s")

    # Verify correctness: check that tables have reasonable values
    total_ctx = sum(t.sum() for t in ppmd2.ctx_tables)
    total_full = sum(t.sum() for t in ppmd2.full_tables)
    print(f"\nTable stats: ctx_total={total_ctx:,}, full_total={total_full:,}")


if __name__ == "__main__":
    benchmark()
