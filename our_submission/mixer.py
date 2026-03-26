from __future__ import annotations
import numpy as np


def logistic_mix(p_neural, p_ngram, alpha, eps=1e-7):
    p_n = np.clip(p_neural, eps, 1.0 - eps)
    p_g = np.clip(p_ngram, eps, 1.0 - eps)
    logit_n = np.log(p_n / (1.0 - p_n))
    logit_g = np.log(p_g / (1.0 - p_g))
    logit_mixed = (1.0 - alpha) * logit_n + alpha * logit_g
    return 1.0 / (1.0 + np.exp(-logit_mixed))


def entropy_adaptive_alpha(ent, base=0.05, range_=0.55, scale=2.0, thresh=4.0):
    return base + range_ / (1.0 + np.exp(-scale * (ent - thresh)))


class PPMDMixer:
    def __init__(self, max_order=7, min_order=2, num_buckets=4_194_304,
                 min_count=2, primes=None):
        if primes is None:
            primes = [36313, 27191, 51647, 81929, 131071, 175447, 209591]
        assert max_order <= len(primes), f"max_order {max_order} > {len(primes)} primes"
        assert num_buckets & (num_buckets - 1) == 0
        self.max_order = max_order
        self.min_order = min_order
        self.num_buckets = num_buckets
        self.min_count = min_count
        self.mask = np.uint64(num_buckets - 1)
        self.n_orders = max_order - min_order + 1

        self.primes = np.array(primes[:max_order], dtype=np.uint64)

        self.ctx_tables = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(self.n_orders)]
        self.full_tables = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(self.n_orders)]
        self.unique_tables = [np.zeros(num_buckets, dtype=np.uint16) for _ in range(self.n_orders)]

    def predict_blended(self, val_np, global_j, n_seg):
        order_p = np.full((self.n_orders, n_seg), -1.0)
        order_w = np.zeros((self.n_orders, n_seg))

        for oi in range(self.n_orders):
            order = self.min_order + oi
            ctx_len = order - 1
            valid = global_j >= ctx_len
            if not valid.any():
                continue
            vi = np.nonzero(valid)[0]

            ck = np.zeros(len(vi), dtype=np.uint64)
            for k in range(ctx_len):
                ti = np.clip(global_j[vi] - ctx_len + k, 0, len(val_np) - 1)
                ck ^= self.primes[k] * val_np[ti].astype(np.uint64)
            ck &= self.mask

            ti = np.clip(global_j[vi], 0, len(val_np) - 1)
            pidx = min(ctx_len, len(self.primes) - 1)
            fk = ck ^ (self.primes[pidx] * val_np[ti].astype(np.uint64))
            fk &= self.mask

            cc = self.ctx_tables[oi][ck].astype(np.float64)
            fc = self.full_tables[oi][fk].astype(np.float64)
            uc = self.unique_tables[oi][ck].astype(np.float64)

            got = cc >= float(self.min_count)
            if got.any():
                mi = vi[got]
                p = np.minimum(fc[got], cc[got]) / np.maximum(cc[got], 1.0)
                order_p[oi, mi] = np.clip(p, 0.0, 1.0)
                esc = uc[got] / (cc[got] + uc[got] + 1e-10)
                order_w[oi, mi] = 1.0 - esc

        has = np.any(order_p >= 0, axis=0)
        out = np.zeros(n_seg)
        if has.any():
            mi = np.nonzero(has)[0]
            tw = np.zeros(len(mi))
            wp = np.zeros(len(mi))
            for oi in range(self.n_orders - 1, -1, -1):
                ok = order_p[oi, mi] >= 0
                if ok.any():
                    w = order_w[oi, mi[ok]] * (2.0 ** oi)
                    wp[ok] += w * order_p[oi, mi[ok]]
                    tw[ok] += w
            out[mi] = wp / np.maximum(tw, 1e-10)
        return out, has

    def update_tables(self, val_np, start, end):
        # must be called AFTER scoring the segment
        for j in range(start, end):
            nt = val_np[j]
            for oi in range(self.n_orders):
                ctx_len = self.min_order + oi - 1
                if j < ctx_len:
                    continue
                ck = np.uint64(0)
                for k in range(ctx_len):
                    ck ^= self.primes[k] * np.uint64(val_np[j - ctx_len + k])
                ck &= self.mask
                pidx = min(ctx_len, len(self.primes) - 1)
                fk = ck ^ (self.primes[pidx] * np.uint64(nt))
                fk &= self.mask
                if self.full_tables[oi][fk] == 0:
                    self.unique_tables[oi][ck] = min(65535, self.unique_tables[oi][ck] + 1)
                self.ctx_tables[oi][ck] += 1
                self.full_tables[oi][fk] += 1
