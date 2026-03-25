"""
Profile PPM-D predict_blended to check if vectorized numpy hash computation
is fast enough for 62M tokens.
"""

from __future__ import annotations

import time
import cProfile
import pstats
from io import StringIO

import numpy as np

from mixer import PPMDMixer

VOCAB_SIZE = 1024


def profile_predict_blended():
    """Profile the hot path in PPM-D prediction."""
    np.random.seed(42)

    # Simulate 100K tokens of pre-existing data
    n_pretrain = 100_000
    tokens = np.random.randint(0, VOCAB_SIZE, size=n_pretrain, dtype=np.int64)

    ppm = PPMDMixer(max_order=7, min_order=2, num_buckets=4_194_304, min_count=2)

    # Pre-fill tables with some data (simulate having seen 50K tokens)
    print("Pre-filling tables with 50K tokens...")
    for j in range(7, 50_000):
        next_token = tokens[j]
        for oi in range(ppm.n_orders):
            order = ppm.min_order + oi
            ctx_len = order - 1
            if j < order:
                continue
            ctx_key = np.uint64(0)
            for k in range(ctx_len):
                ctx_key ^= ppm.primes[k] * np.uint64(tokens[j - 1 - k])
            ctx_key &= ppm.mask
            prime_idx = min(ctx_len, len(ppm.primes) - 1)
            full_key = ctx_key ^ (ppm.primes[prime_idx] * np.uint64(next_token))
            full_key &= ppm.mask
            if ppm.full_tables[oi][full_key] == 0:
                ppm.unique_tables[oi][ctx_key] = min(65535, ppm.unique_tables[oi][ctx_key] + 1)
            ppm.ctx_tables[oi][ctx_key] += 1
            ppm.full_tables[oi][full_key] += 1

    # Now profile predict_blended on segments
    stride = 64
    seq_len = 1024
    segment_sizes = []
    segment_times = []

    print("\nProfiling predict_blended on 50K tokens...")
    total_scored = 0
    t0 = time.perf_counter()

    for ws in range(50_000, n_pretrain - seq_len, stride):
        wlen = min(seq_len, n_pretrain - 1 - ws)
        s = max(wlen - stride, 0)
        seg_len = wlen - s
        if seg_len <= 0:
            continue

        global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)

        t_seg = time.perf_counter()
        p_blend, has_match = ppm.predict_blended(tokens, global_j, seg_len)
        dt = time.perf_counter() - t_seg

        segment_sizes.append(seg_len)
        segment_times.append(dt)
        total_scored += seg_len

        # Update tables (score-first)
        ppm.update_tables(tokens, ws + s + 1, min(ws + wlen + 1, len(tokens)))

    elapsed = time.perf_counter() - t0

    # Analysis
    segment_times = np.array(segment_times)
    segment_sizes = np.array(segment_sizes)

    rate = total_scored / elapsed
    print(f"\nResults:")
    print(f"  Total tokens scored: {total_scored:,}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Rate: {rate:,.0f} tok/s")
    print(f"  Per-segment avg: {segment_times.mean()*1e6:.1f} µs")
    print(f"  Per-segment p99: {np.percentile(segment_times, 99)*1e6:.1f} µs")
    print(f"  Per-segment max: {segment_times.max()*1e6:.1f} µs")
    print(f"\n  Extrapolated 62M tokens: {62_000_000 / rate:.1f}s ({62_000_000 / rate / 60:.1f} min)")

    # Bottleneck analysis: cProfile the predict_blended call
    print("\n--- cProfile on 1000 predict_blended calls ---")
    pr = cProfile.Profile()
    pr.enable()
    for ws in range(50_000, 50_000 + 1000 * stride, stride):
        wlen = min(seq_len, n_pretrain - 1 - ws)
        s = max(wlen - stride, 0)
        seg_len = wlen - s
        if seg_len <= 0:
            continue
        global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
        ppm.predict_blended(tokens, global_j, seg_len)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    # Memory usage
    mem_mb = sum(
        a.nbytes for tables in [ppm.ctx_tables, ppm.full_tables, ppm.unique_tables] for a in tables
    ) / (1024 * 1024)
    print(f"PPM-D memory: {mem_mb:.1f} MB")

    # Breakdown: where time goes in predict_blended
    print("\n--- Timing breakdown (1 call) ---")
    ws = 60_000
    wlen = min(seq_len, n_pretrain - 1 - ws)
    s = max(wlen - stride, 0)
    seg_len = wlen - s
    global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
    n_seg = seg_len

    # Hash computation timing
    t1 = time.perf_counter()
    for _ in range(100):
        order_p = np.full((ppm.n_orders, n_seg), -1.0)
        for oi in range(ppm.n_orders):
            order = ppm.min_order + oi
            ctx_len = order - 1
            valid_mask = global_j >= order
            if not valid_mask.any():
                continue
            v_idx = np.nonzero(valid_mask)[0]
            ctx_keys = np.zeros(len(v_idx), dtype=np.uint64)
            for k in range(ctx_len):
                tok_idx = global_j[v_idx] - 1 - k
                tok_idx = np.clip(tok_idx, 0, len(tokens) - 1)
                ctx_keys ^= ppm.primes[k] * tokens[tok_idx].astype(np.uint64)
            ctx_keys &= ppm.mask
    t2 = time.perf_counter()
    print(f"  Hash computation (100 iter): {(t2-t1)*1e3:.1f} ms ({(t2-t1)/100*1e6:.1f} µs/call)")

    # Table lookup timing
    t1 = time.perf_counter()
    for _ in range(100):
        for oi in range(ppm.n_orders):
            order = ppm.min_order + oi
            ctx_len = order - 1
            valid_mask = global_j >= order
            if not valid_mask.any():
                continue
            v_idx = np.nonzero(valid_mask)[0]
            ctx_keys = np.zeros(len(v_idx), dtype=np.uint64)
            for k in range(ctx_len):
                tok_idx = global_j[v_idx] - 1 - k
                tok_idx = np.clip(tok_idx, 0, len(tokens) - 1)
                ctx_keys ^= ppm.primes[k] * tokens[tok_idx].astype(np.uint64)
            ctx_keys &= ppm.mask
            ctx_counts = ppm.ctx_tables[oi][ctx_keys]
    t2 = time.perf_counter()
    print(f"  Hash + lookup (100 iter):    {(t2-t1)*1e3:.1f} ms ({(t2-t1)/100*1e6:.1f} µs/call)")

    # Update timing
    t1 = time.perf_counter()
    dummy_ppm = PPMDMixer(max_order=7, min_order=2, num_buckets=4_194_304)
    for _ in range(10):
        dummy_ppm.update_tables(tokens, 50_000, 50_064)
    t2 = time.perf_counter()
    print(f"  update_tables 64 tok (10x):  {(t2-t1)*1e3:.1f} ms ({(t2-t1)/10*1e3:.1f} ms/call)")
    print(f"  → update is Python loops, potential Numba target")


if __name__ == "__main__":
    profile_predict_blended()
