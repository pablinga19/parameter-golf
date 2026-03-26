from __future__ import annotations
import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

PRIMES = np.array([36313, 27191, 51647, 81929, 131071, 175447, 209591], dtype=np.uint64)


@njit(cache=True)
def ppmd_update_batch(val_np, start, end, n_orders, min_order, primes, mask,
                      ctx_tables, full_tables, unique_tables):
    for j in range(start, end):
        nt = np.uint64(val_np[j])
        for oi in range(n_orders):
            ctx_len = min_order + oi - 1
            if j < ctx_len:
                continue
            ck = np.uint64(0)
            for k in range(ctx_len):
                ck ^= primes[k] * np.uint64(val_np[j - ctx_len + k])
            ck &= mask
            pidx = min(ctx_len, len(primes) - 1)
            fk = ck ^ (primes[pidx] * nt)
            fk &= mask
            if full_tables[oi][fk] == 0:
                if unique_tables[oi][ck] < 65535:
                    unique_tables[oi][ck] += 1
            ctx_tables[oi][ck] += 1
            full_tables[oi][fk] += 1


@njit(cache=True, parallel=True)
def ppmd_predict_batch(val_np, global_j, n_seg, n_orders, min_order, min_count,
                       primes, mask, ctx_tables, full_tables, unique_tables,
                       depth_boost):
    p_out = np.zeros(n_seg, dtype=np.float64)
    has = np.zeros(n_seg, dtype=np.bool_)

    for idx in prange(n_seg):
        j = global_j[idx]
        nt = np.uint64(val_np[j])
        tw = 0.0
        wp = 0.0
        found = False

        for oi in range(n_orders - 1, -1, -1):
            ctx_len = min_order + oi - 1
            if j < ctx_len:
                continue
            ck = np.uint64(0)
            for k in range(ctx_len):
                ck ^= primes[k] * np.uint64(val_np[j - ctx_len + k])
            ck &= mask
            cc = float(ctx_tables[oi][ck])
            if cc < float(min_count):
                continue
            pidx = min(ctx_len, len(primes) - 1)
            fk = ck ^ (primes[pidx] * nt)
            fk &= mask
            fc = float(full_tables[oi][fk])
            uc = float(unique_tables[oi][ck])

            p = min(fc, cc) / max(cc, 1.0)
            p = max(0.0, min(1.0, p))
            esc = uc / (cc + uc + 1e-10)
            w = (1.0 - esc) * depth_boost[oi]
            wp += w * p
            tw += w
            found = True

        if found and tw > 1e-10:
            p_out[idx] = wp / tw
            has[idx] = True

    return p_out, has


class PPMDNumba:
    def __init__(self, max_order=7, min_order=2, num_buckets=4_194_304,
                 min_count=2, depth_boost_base=2.0):
        assert num_buckets & (num_buckets - 1) == 0
        self.max_order = max_order
        self.min_order = min_order
        self.num_buckets = num_buckets
        self.min_count = min_count
        self.n_orders = max_order - min_order + 1
        self.mask = np.uint64(num_buckets - 1)
        self.primes = PRIMES[:max_order].copy()

        self.ctx_tables = NumbaList()
        self.full_tables = NumbaList()
        self.unique_tables = NumbaList()
        for _ in range(self.n_orders):
            self.ctx_tables.append(np.zeros(num_buckets, dtype=np.uint32))
            self.full_tables.append(np.zeros(num_buckets, dtype=np.uint32))
            self.unique_tables.append(np.zeros(num_buckets, dtype=np.uint16))

        self.depth_boost = np.array(
            [depth_boost_base ** i for i in range(self.n_orders)], dtype=np.float64)

    def predict_blended(self, val_np, global_j, n_seg):
        return ppmd_predict_batch(
            val_np, global_j, n_seg, self.n_orders, self.min_order,
            self.min_count, self.primes, self.mask,
            self.ctx_tables, self.full_tables, self.unique_tables,
            self.depth_boost)

    def update_tables(self, val_np, start, end):
        ppmd_update_batch(
            val_np, start, end, self.n_orders, self.min_order,
            self.primes, self.mask,
            self.ctx_tables, self.full_tables, self.unique_tables)
