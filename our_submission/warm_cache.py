import numpy as np
import struct
import zstandard


def build_cache_from_corpus(token_files, max_order=7, min_order=2,
                            num_buckets=4_194_304):
    """pre-compute n-gram statistics from training corpus.

    reads training token files and builds the same hash tables
    used by the eval-time n-gram cache. the tables can then be
    compressed and stored in the 16MB artifact, giving the cache
    a warm start at eval time instead of cold.
    """
    primes = np.array([36313, 27191, 51647, 81929, 131071, 175447, 209591],
                      dtype=np.uint64)
    mask = np.uint64(num_buckets - 1)
    n_orders = max_order - min_order + 1

    # uint64 during build to avoid overflow, downcast to uint32 after
    ctx_tables = [np.zeros(num_buckets, dtype=np.uint64) for _ in range(n_orders)]
    full_tables = [np.zeros(num_buckets, dtype=np.uint64) for _ in range(n_orders)]

    total_tokens = 0
    for fpath in token_files:
        tokens = np.fromfile(fpath, dtype=np.uint16)
        for j in range(min_order, len(tokens)):
            for oi in range(n_orders):
                ctx_len = min_order + oi - 1
                ck = np.uint64(0)
                for k in range(ctx_len):
                    ck ^= primes[k] * np.uint64(tokens[j - ctx_len + k])
                ck &= mask
                pidx = min(ctx_len, len(primes) - 1)
                fk = ck ^ (primes[pidx] * np.uint64(tokens[j]))
                fk &= mask
                ctx_tables[oi][int(ck)] += 1
                full_tables[oi][int(fk)] += 1
        total_tokens += len(tokens)

    # clip to uint32 range for storage
    ctx_tables = [np.minimum(ct, np.iinfo(np.uint32).max).astype(np.uint32) for ct in ctx_tables]
    full_tables = [np.minimum(ft, np.iinfo(np.uint32).max).astype(np.uint32) for ft in full_tables]
    return ctx_tables, full_tables, total_tokens


def compress_cache(ctx_tables, full_tables, level=19):
    """compress the warm cache tables with zstd.

    returns compressed bytes and the uncompressed size for budgeting.
    """
    parts = []
    for ct, ft in zip(ctx_tables, full_tables):
        parts.append(ct.tobytes())
        parts.append(ft.tobytes())
    raw = b''.join(parts)

    cctx = zstandard.ZstdCompressor(level=level)
    compressed = cctx.compress(raw)

    return compressed, len(raw)


def decompress_cache(compressed, n_orders, num_buckets):
    """decompress warm cache back to numpy tables."""
    dctx = zstandard.ZstdDecompressor()
    raw = dctx.decompress(compressed)

    bytes_per_table = num_buckets * 4  # uint32
    ctx_tables = []
    full_tables = []
    offset = 0
    for _ in range(n_orders):
        ct = np.frombuffer(raw[offset:offset + bytes_per_table], dtype=np.uint32).copy()
        offset += bytes_per_table
        ft = np.frombuffer(raw[offset:offset + bytes_per_table], dtype=np.uint32).copy()
        offset += bytes_per_table
        ctx_tables.append(ct)
        full_tables.append(ft)

    return ctx_tables, full_tables


def estimate_cache_size(n_orders=6, num_buckets=4_194_304):
    """estimate compressed size of warm cache.

    most buckets will be zero (sparse), so zstd compresses well.
    """
    # worst case: all buckets populated (dense)
    raw_bytes = n_orders * 2 * num_buckets * 4
    # typical: ~5-10% of buckets non-zero -> ~10-20x compression
    estimated = raw_bytes // 15
    return {
        'raw_bytes': raw_bytes,
        'estimated_compressed': estimated,
        'raw_mb': raw_bytes / 1e6,
        'estimated_mb': estimated / 1e6,
    }


if __name__ == "__main__":
    est = estimate_cache_size()
    print(f"warm cache estimate:")
    print(f"  raw: {est['raw_mb']:.1f} MB")
    print(f"  compressed (est): {est['estimated_mb']:.1f} MB")
    print(f"  budget impact: {est['estimated_mb']:.1f} MB of 16 MB")

    # test roundtrip with synthetic data
    n_orders = 6
    num_buckets = 4_194_304
    ctx = [np.random.randint(0, 100, num_buckets, dtype=np.uint32) for _ in range(n_orders)]
    full = [np.random.randint(0, 50, num_buckets, dtype=np.uint32) for _ in range(n_orders)]

    comp, raw_size = compress_cache(ctx, full)
    print(f"\nroundtrip test (random data):")
    print(f"  raw: {raw_size/1e6:.1f} MB  compressed: {len(comp)/1e6:.1f} MB  ratio: {len(comp)/raw_size:.3f}")

    ctx2, full2 = decompress_cache(comp, n_orders, num_buckets)
    for i in range(n_orders):
        assert np.array_equal(ctx[i], ctx2[i])
        assert np.array_equal(full[i], full2[i])
    print("  roundtrip: ok")

    # test with sparse data (realistic — most buckets empty)
    ctx_sparse = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(n_orders)]
    full_sparse = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(n_orders)]
    # populate 5% of buckets
    for i in range(n_orders):
        idx = np.random.choice(num_buckets, size=num_buckets // 20, replace=False)
        ctx_sparse[i][idx] = np.random.randint(1, 100, len(idx), dtype=np.uint32)
        full_sparse[i][idx] = np.random.randint(1, 50, len(idx), dtype=np.uint32)

    comp_sparse, raw_sparse = compress_cache(ctx_sparse, full_sparse)
    print(f"\nsparse test (5% populated):")
    print(f"  raw: {raw_sparse/1e6:.1f} MB  compressed: {len(comp_sparse)/1e6:.1f} MB  ratio: {len(comp_sparse)/raw_sparse:.3f}")
    print(f"  fits in 16MB budget: {'YES' if len(comp_sparse) < 4_000_000 else 'NO'} ({len(comp_sparse)/1e6:.1f} MB)")
