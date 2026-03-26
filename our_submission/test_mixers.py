"""
Integration test: compare mixing strategies on real FineWeb validation data.

Runs the SOTA linear backoff, our logistic mixer, and PPM-D blending
on the first N tokens of FineWeb validation data, comparing n-gram
prediction quality and mixed BPB.

Since we don't have a trained neural model, we simulate neural predictions
as uniform (1/vocab_size) — this isolates the n-gram contribution.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

# Our mixers
from mixer import (
    PPMDMixer,
    logistic_mix,
    entropy_adaptive_alpha,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets" / "fineweb10B_sp1024"
VOCAB_SIZE = 1024
MAX_TOKENS = 100_000  # first 100K tokens for the test


def load_val_tokens(max_tokens: int = MAX_TOKENS) -> np.ndarray:
    """Load validation tokens from the binary shard."""
    val_path = DATA_DIR / "fineweb_val_000000.bin"
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {val_path}. "
            "Run: python3 data/cached_challenge_fineweb.py --variant sp1024"
        )
    # Shards are uint16 tokens stored as raw binary
    tokens = np.fromfile(val_path, dtype=np.uint16)
    tokens = tokens[:max_tokens].astype(np.int64)
    print(f"Loaded {len(tokens):,} tokens from {val_path.name}")
    return tokens


# ---------------------------------------------------------------------------
# SOTA: Multi-order backoff with linear mixing
# ---------------------------------------------------------------------------

def run_sota_backoff(
    val_np: np.ndarray,
    stride: int = 64,
    seq_len: int = 1024,
    min_order: int = 2,
    max_order: int = 7,
    num_buckets: int = 4_194_304,
    min_count: int = 2,
    mix_method: str = "linear",
) -> dict:
    """Replicate the SOTA sliding eval n-gram protocol.

    mix_method: "linear" (SOTA), "logistic" (ours), "none" (n-gram only)
    """
    n_orders = max_order - min_order + 1
    mask = np.uint64(num_buckets - 1)
    primes = np.array(
        [36313, 27191, 51647, 81929, 131071, 175447, 209591], dtype=np.uint64
    )

    ctx_tables = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(n_orders)]
    full_tables = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(n_orders)]

    total_tokens = len(val_np) - 1
    # Simulated model: uniform probability = 1/vocab_size
    uniform_p = 1.0 / VOCAB_SIZE

    nll_sum = 0.0
    nll_sum_model_only = 0.0
    token_count = 0
    ngram_hit_count = 0
    ngram_nll_sum = 0.0

    t0 = time.perf_counter()

    for ws in range(0, total_tokens, stride):
        wlen = min(seq_len, total_tokens - ws)
        if wlen < 1:
            continue
        s = 0 if ws == 0 else max(wlen - stride, 0)
        seg_len = wlen - s
        if seg_len <= 0:
            continue

        n_seg = seg_len
        global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)

        # Simulated model probabilities (uniform)
        seg_model_p = np.full(n_seg, uniform_p)
        # Simulated entropy (max entropy for uniform distribution)
        seg_entropy = np.full(n_seg, math.log(VOCAB_SIZE))

        # --- Multi-order backoff (exact SOTA protocol) ---
        order_data = []
        for oi in range(n_orders):
            ctx_w = min_order + oi - 1  # context width
            valid = global_j >= ctx_w
            if not valid.any():
                order_data.append(None)
                continue
            v_idx = np.nonzero(valid)[0]
            jv = global_j[v_idx]
            ctx_hash = np.zeros(len(jv), dtype=np.uint64)
            for k in range(ctx_w):
                tok = val_np[jv - (ctx_w - k)].astype(np.uint64)
                ctx_hash ^= tok * primes[k % len(primes)]
            ctx_key = (ctx_hash & mask).astype(np.int64)
            tgt_np = val_np[jv].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt_np * primes[ctx_w % len(primes)])) & mask).astype(np.int64)
            order_data.append((v_idx, ctx_key, full_key))

        # Highest order first backoff
        best_p_ng = np.full(n_seg, -1.0)
        for oi in range(n_orders - 1, -1, -1):
            if order_data[oi] is None:
                continue
            v_idx, ctx_key, full_key = order_data[oi]
            ctx_counts = ctx_tables[oi][ctx_key].astype(np.float64)
            full_counts = full_tables[oi][full_key].astype(np.float64)
            has_match = ctx_counts >= float(min_count)
            needs_fill = has_match & (best_p_ng[v_idx] < 0)
            if needs_fill.any():
                fill_idx = v_idx[needs_fill]
                p = np.minimum(full_counts[needs_fill], ctx_counts[needs_fill]) / np.maximum(ctx_counts[needs_fill], 1.0)
                best_p_ng[fill_idx] = np.clip(p, 0.0, 1.0)

        # --- Mix ---
        has_match = best_p_ng >= 0
        result_p = seg_model_p.copy()

        if has_match.any():
            alpha = entropy_adaptive_alpha(seg_entropy[has_match])

            if mix_method == "linear":
                result_p[has_match] = (1.0 - alpha) * seg_model_p[has_match] + alpha * best_p_ng[has_match]
            elif mix_method == "logistic":
                result_p[has_match] = logistic_mix(
                    seg_model_p[has_match], best_p_ng[has_match], alpha
                )
            elif mix_method == "none":
                # Pure n-gram (where available)
                result_p[has_match] = best_p_ng[has_match]

            ngram_hit_count += has_match.sum()
            ngram_nll_sum += (-np.log(np.clip(best_p_ng[has_match], 1e-12, 1.0))).sum()

        nll = -np.log(np.clip(result_p, 1e-12, 1.0))
        nll_sum += nll.sum()
        nll_sum_model_only += (-np.log(np.clip(seg_model_p, 1e-12, 1.0))).sum()
        token_count += n_seg

        # Score-first: update AFTER scoring
        for oi in range(n_orders):
            if order_data[oi] is None:
                continue
            v_idx, ctx_key, full_key = order_data[oi]
            np.add.at(ctx_tables[oi], ctx_key, 1)
            np.add.at(full_tables[oi], full_key, 1)

    elapsed = time.perf_counter() - t0

    avg_nll = nll_sum / token_count
    avg_bpb_approx = avg_nll / math.log(2)  # bits per token (approx BPB for sp1024)
    model_only_bpb = nll_sum_model_only / token_count / math.log(2)
    ngram_avg_nll = ngram_nll_sum / max(ngram_hit_count, 1)

    return {
        "method": mix_method,
        "tokens": token_count,
        "avg_nll": avg_nll,
        "avg_bpb": avg_bpb_approx,
        "model_only_bpb": model_only_bpb,
        "ngram_hits": int(ngram_hit_count),
        "ngram_hit_rate": ngram_hit_count / token_count,
        "ngram_avg_nll": ngram_avg_nll,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# PPM-D blended evaluation
# ---------------------------------------------------------------------------

def run_ppmd_blended(
    val_np: np.ndarray,
    stride: int = 64,
    seq_len: int = 1024,
    mix_method: str = "logistic",
) -> dict:
    """Run PPM-D blended prediction on validation tokens."""
    ppm = PPMDMixer(max_order=7, min_order=2, num_buckets=4_194_304, min_count=2)

    total_tokens = len(val_np) - 1
    uniform_p = 1.0 / VOCAB_SIZE

    nll_sum = 0.0
    token_count = 0
    ngram_hit_count = 0

    t0 = time.perf_counter()

    for ws in range(0, total_tokens, stride):
        wlen = min(seq_len, total_tokens - ws)
        if wlen < 1:
            continue
        s = 0 if ws == 0 else max(wlen - stride, 0)
        seg_len = wlen - s
        if seg_len <= 0:
            continue

        n_seg = seg_len
        global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)

        seg_model_p = np.full(n_seg, uniform_p)
        seg_entropy = np.full(n_seg, math.log(VOCAB_SIZE))

        # PPM-D blended prediction
        p_blend, has_match = ppm.predict_blended(val_np, global_j, n_seg)

        result_p = seg_model_p.copy()
        if has_match.any():
            alpha = entropy_adaptive_alpha(seg_entropy[has_match])
            if mix_method == "logistic":
                result_p[has_match] = logistic_mix(
                    seg_model_p[has_match], p_blend[has_match], alpha
                )
            else:
                result_p[has_match] = (1.0 - alpha) * seg_model_p[has_match] + alpha * p_blend[has_match]
            ngram_hit_count += has_match.sum()

        nll = -np.log(np.clip(result_p, 1e-12, 1.0))
        nll_sum += nll.sum()
        token_count += n_seg

        # Score-first update
        scored_start = ws + s + 1
        scored_end = ws + wlen + 1
        ppm.update_tables(val_np, scored_start, min(scored_end, len(val_np)))

    elapsed = time.perf_counter() - t0

    avg_nll = nll_sum / token_count
    avg_bpb = avg_nll / math.log(2)

    return {
        "method": f"ppmd+{mix_method}",
        "tokens": token_count,
        "avg_nll": avg_nll,
        "avg_bpb": avg_bpb,
        "ngram_hits": int(ngram_hit_count),
        "ngram_hit_rate": ngram_hit_count / token_count,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CTW evaluation (optional — slow without Numba)
# ---------------------------------------------------------------------------

def run_ctw(
    val_np: np.ndarray,
    stride: int = 64,
    seq_len: int = 1024,
    mix_method: str = "logistic",
    use_numba: bool = True,
    max_tokens: int | None = None,
) -> dict:
    """Run CTW prediction. Uses Numba version if available."""
    if use_numba:
        from ctw_numba import NumbaCTWMixer, warmup_jit
        print("Warming up Numba JIT...")
        warmup_jit()
        ctw = NumbaCTWMixer(max_depth=7, vocab_size=VOCAB_SIZE, num_buckets=1_048_576)
        print(f"CTW memory: {ctw.memory_mb():.1f} MB")
    else:
        from mixer import CTWMixer
        ctw = CTWMixer(max_depth=7, vocab_size=VOCAB_SIZE, num_buckets=1_048_576)

    effective_tokens = len(val_np) - 1
    if max_tokens:
        effective_tokens = min(effective_tokens, max_tokens)
    uniform_p = 1.0 / VOCAB_SIZE

    nll_sum = 0.0
    token_count = 0

    t0 = time.perf_counter()

    for ws in range(0, effective_tokens, stride):
        wlen = min(seq_len, effective_tokens - ws)
        if wlen < 1:
            continue
        s = 0 if ws == 0 else max(wlen - stride, 0)
        seg_len = wlen - s
        if seg_len <= 0:
            continue

        n_seg = seg_len
        global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
        seg_model_p = np.full(n_seg, uniform_p)
        seg_entropy = np.full(n_seg, math.log(VOCAB_SIZE))

        p_ctw, has_match = ctw.predict_batch(val_np, global_j, n_seg)

        result_p = seg_model_p.copy()
        if has_match.any():
            alpha = entropy_adaptive_alpha(seg_entropy[has_match])
            if mix_method == "logistic":
                result_p[has_match] = logistic_mix(
                    seg_model_p[has_match], p_ctw[has_match], alpha
                )
            else:
                result_p[has_match] = (1.0 - alpha) * seg_model_p[has_match] + alpha * p_ctw[has_match]

        nll = -np.log(np.clip(result_p, 1e-12, 1.0))
        nll_sum += nll.sum()
        token_count += n_seg

        scored_start = ws + s + 1
        scored_end = ws + wlen + 1
        ctw.update_batch(val_np, scored_start, min(scored_end, len(val_np)))

    elapsed = time.perf_counter() - t0

    avg_nll = nll_sum / token_count
    avg_bpb = avg_nll / math.log(2)

    return {
        "method": f"ctw_{'numba' if use_numba else 'python'}+{mix_method}",
        "tokens": token_count,
        "avg_nll": avg_nll,
        "avg_bpb": avg_bpb,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Parameter Golf — Mixer Strategy Comparison on FineWeb Validation Data")
    print("=" * 80)

    val_np = load_val_tokens(MAX_TOKENS)
    print(f"Vocab range: [{val_np.min()}, {val_np.max()}]")
    print()

    results = []

    # 1. SOTA: linear backoff
    print("--- SOTA: Linear backoff mixing ---")
    r = run_sota_backoff(val_np, mix_method="linear")
    results.append(r)
    print(f"  BPB: {r['avg_bpb']:.6f} | hits: {r['ngram_hits']:,} ({r['ngram_hit_rate']:.1%}) | {r['elapsed_s']:.2f}s")

    # 2. Our logistic backoff (same n-gram data, different mixing)
    print("--- Ours: Logistic backoff mixing ---")
    r = run_sota_backoff(val_np, mix_method="logistic")
    results.append(r)
    print(f"  BPB: {r['avg_bpb']:.6f} | hits: {r['ngram_hits']:,} ({r['ngram_hit_rate']:.1%}) | {r['elapsed_s']:.2f}s")

    # 3. PPM-D blended + logistic
    print("--- Ours: PPM-D blended + logistic ---")
    r = run_ppmd_blended(val_np, mix_method="logistic")
    results.append(r)
    print(f"  BPB: {r['avg_bpb']:.6f} | hits: {r['ngram_hits']:,} ({r['ngram_hit_rate']:.1%}) | {r['elapsed_s']:.2f}s")

    # 4. PPM-D blended + linear (to isolate blending vs mixing effect)
    print("--- PPM-D blended + linear ---")
    r = run_ppmd_blended(val_np, mix_method="linear")
    results.append(r)
    print(f"  BPB: {r['avg_bpb']:.6f} | hits: {r['ngram_hits']:,} ({r['ngram_hit_rate']:.1%}) | {r['elapsed_s']:.2f}s")

    # 5. CTW (Numba) + logistic — full 100K tokens (v2 is fast enough)
    print("--- CTW (Numba) + logistic ---")
    try:
        r = run_ctw(val_np, mix_method="logistic", use_numba=True)
        results.append(r)
        print(f"  BPB: {r['avg_bpb']:.6f} | {r['elapsed_s']:.2f}s")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # 6. CTW (Numba) + linear
    print("--- CTW (Numba) + linear ---")
    try:
        r = run_ctw(val_np, mix_method="linear", use_numba=True)
        results.append(r)
        print(f"  BPB: {r['avg_bpb']:.6f} | {r['elapsed_s']:.2f}s")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # Summary table
    print()
    print("=" * 80)
    print(f"{'Method':<35} {'BPB':>10} {'NLL':>10} {'Hits':>10} {'Time':>8}")
    print("-" * 80)
    baseline_bpb = results[0]["avg_bpb"]
    for r in results:
        delta = r["avg_bpb"] - baseline_bpb
        delta_str = f"({delta:+.4f})" if r["method"] != "linear" else ""
        hits_str = f"{r.get('ngram_hits', 'N/A'):>10,}" if isinstance(r.get('ngram_hits'), int) else f"{'N/A':>10}"
        print(f"{r['method']:<35} {r['avg_bpb']:>10.6f} {r['avg_nll']:>10.6f} {hits_str} {r['elapsed_s']:>7.2f}s {delta_str}")

    print()
    print("NOTE: These results use UNIFORM model probs (1/1024). With a real neural")
    print("model, the absolute BPB is much lower, but the RELATIVE differences between")
    print("mixing strategies should hold or amplify (logistic shines more at extremes).")


if __name__ == "__main__":
    main()
