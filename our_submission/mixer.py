"""Probability mixers for n-gram eval cache."""

from __future__ import annotations
import math
from typing import Optional
import numpy as np


def logistic_mix(p_neural, p_ngram, alpha, eps=1e-7):
    """Mix probabilities in log-odds space."""
    p_n = np.clip(p_neural, eps, 1.0 - eps)
    p_g = np.clip(p_ngram, eps, 1.0 - eps)
    logit_n = np.log(p_n / (1.0 - p_n))
    logit_g = np.log(p_g / (1.0 - p_g))
    logit_mixed = (1.0 - alpha) * logit_n + alpha * logit_g
    return 1.0 / (1.0 + np.exp(-logit_mixed))


def entropy_adaptive_alpha(ent, base=0.05, range_=0.55, scale=2.0, thresh=4.0):
    return base + range_ / (1.0 + np.exp(-scale * (ent - thresh)))


class PPMDMixer:
    """PPM-D with blended orders instead of pure backoff.

    Escape weight per order: unique / (total + unique).
    Higher orders boosted exponentially.
    """

    def __init__(self, max_order=7, min_order=2, num_buckets=4_194_304,
                 min_count=2, primes=None):
        assert num_buckets & (num_buckets - 1) == 0, "buckets must be power of 2"
        self.max_order = max_order
        self.min_order = min_order
        self.num_buckets = num_buckets
        self.min_count = min_count
        self.mask = np.uint64(num_buckets - 1)
        self.n_orders = max_order - min_order + 1

        if primes is None:
            primes = [36313, 27191, 51647, 81929, 131071, 175447, 209591]
        self.primes = np.array(primes[:max_order], dtype=np.uint64)

        self.ctx_tables = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(self.n_orders)]
        self.full_tables = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(self.n_orders)]
        self.unique_tables = [np.zeros(num_buckets, dtype=np.uint16) for _ in range(self.n_orders)]

    def _hash_ctx(self, tokens, pos, order):
        h = np.uint64(0)
        for k in range(order):
            idx = pos - 1 - k
            if idx < 0:
                return np.uint64(0xFFFFFFFFFFFFFFFF)
            h ^= self.primes[k] * np.uint64(tokens[idx])
        return h & self.mask

    def _hash_full(self, tokens, pos, next_tok, order):
        h = self._hash_ctx(tokens, pos, order)
        if h == np.uint64(0xFFFFFFFFFFFFFFFF):
            return h
        pidx = min(order, len(self.primes) - 1)
        h ^= self.primes[pidx] * np.uint64(next_tok)
        return h & self.mask

    def predict_blended(self, val_np, global_j, n_seg):
        """Returns (p_blend, has_match) arrays."""
        order_p = np.full((self.n_orders, n_seg), -1.0)
        order_w = np.zeros((self.n_orders, n_seg))

        for oi in range(self.n_orders):
            order = self.min_order + oi
            ctx_len = order - 1
            valid = global_j >= ctx_len
            if not valid.any():
                continue

            vi = np.nonzero(valid)[0]

            # hash preceding tokens only
            ck = np.zeros(len(vi), dtype=np.uint64)
            for k in range(ctx_len):
                ti = np.clip(global_j[vi] - 1 - k, 0, len(val_np) - 1)
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
        """Update counts after scoring [start, end). Must be called AFTER scoring."""
        for j in range(start, end):
            nt = val_np[j]
            for oi in range(self.n_orders):
                ctx_len = self.min_order + oi - 1
                if j < ctx_len:
                    continue
                ck = np.uint64(0)
                for k in range(ctx_len):
                    ck ^= self.primes[k] * np.uint64(val_np[j - 1 - k])
                ck &= self.mask
                pidx = min(ctx_len, len(self.primes) - 1)
                fk = ck ^ (self.primes[pidx] * np.uint64(nt))
                fk &= self.mask

                if self.full_tables[oi][fk] == 0:
                    self.unique_tables[oi][ck] = min(65535, self.unique_tables[oi][ck] + 1)
                self.ctx_tables[oi][ck] += 1
                self.full_tables[oi][fk] += 1


# experimental CTW variant, kept for comparison

class KTEstimator:
    __slots__ = ('counts', 'total', 'vs')
    def __init__(self, vs=1024):
        self.counts = {}
        self.total = 0
        self.vs = vs
    def predict(self, tok):
        return (self.counts.get(tok, 0) + 0.5) / (self.total + self.vs * 0.5)
    def update(self, tok):
        self.counts[tok] = self.counts.get(tok, 0) + 1
        self.total += 1

class HashedCTWNode:
    __slots__ = ('kt', 'log_pe', 'log_pw')
    def __init__(self, vs=1024):
        self.kt = KTEstimator(vs)
        self.log_pe = 0.0
        self.log_pw = 0.0

class CTWMixer:
    """Hashed KT blend across depths."""

    def __init__(self, max_depth=7, vocab_size=1024, num_buckets=1_048_576, primes=None):
        assert num_buckets & (num_buckets - 1) == 0
        self.max_depth = max_depth
        self.vocab_size = vocab_size
        self.num_buckets = num_buckets
        self.mask = num_buckets - 1
        self.primes = primes or [36313, 27191, 51647, 81929, 131071, 175447, 209591]
        self.nodes = [{} for _ in range(max_depth + 1)]

    def _hash(self, ctx, depth):
        h = 0
        for k in range(depth):
            idx = len(ctx) - 1 - k
            if idx < 0:
                return -1
            h ^= self.primes[k] * int(ctx[idx])
        return h & self.mask

    def _node(self, d, hk):
        if hk not in self.nodes[d]:
            self.nodes[d][hk] = HashedCTWNode(self.vocab_size)
        return self.nodes[d][hk]

    def predict(self, ctx, tok):
        preds = []
        for d in range(self.max_depth + 1):
            h = 0 if d == 0 else self._hash(ctx, d)
            if h < 0:
                break
            preds.append(self._node(d, h).kt.predict(tok))
        if not preds:
            return 1.0 / self.vocab_size
        pw = preds[-1]
        for d in range(len(preds) - 2, -1, -1):
            pw = 0.5 * preds[d] + 0.5 * pw
        return pw

    def update(self, ctx, tok):
        for d in range(self.max_depth + 1):
            h = 0 if d == 0 else self._hash(ctx, d)
            if h < 0:
                break
            self._node(d, h).kt.update(tok)

    def predict_batch(self, val_np, global_j, n_seg):
        p = np.zeros(n_seg)
        m = np.ones(n_seg, dtype=bool)
        for idx in range(n_seg):
            j = int(global_j[idx])
            if j <= 0 or j >= len(val_np):
                p[idx] = 1.0 / self.vocab_size
                continue
            p[idx] = self.predict(val_np[max(0, j - self.max_depth):j], int(val_np[j]))
        return p, m

    def update_batch(self, val_np, start, end):
        for j in range(start, end):
            if j <= 0 or j >= len(val_np):
                continue
            self.update(val_np[max(0, j - self.max_depth):j], int(val_np[j]))


def mix_predictions(p_neural, p_classical, has_match, ent,
                    method="logistic", ab=0.05, ar=0.55, asc=2.0, at=4.0):
    """Mix neural + classical predictions. Returns NLL array."""
    out = p_neural.copy()
    if has_match.any():
        alpha = np.clip(entropy_adaptive_alpha(ent[has_match], ab, ar, asc, at), 0.0, 0.95)
        if method == "linear":
            out[has_match] = (1.0 - alpha) * p_neural[has_match] + alpha * p_classical[has_match]
        elif method == "logistic":
            out[has_match] = logistic_mix(p_neural[has_match], p_classical[has_match], alpha)
        elif method == "adaptive":
            boost = np.where(p_classical[has_match] > 0.5, 1.3, 1.0)
            out[has_match] = logistic_mix(p_neural[has_match], p_classical[has_match],
                                          np.clip(alpha * boost, 0.0, 0.95))
    return -np.log(np.clip(out, 1e-12, 1.0))
