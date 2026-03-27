import numpy as np
from collections import Counter


class LZCopyModel:
    """variable-length exact match predictor using rolling hash.

    finds the longest previous occurrence of the current context in
    already-scored tokens, then predicts the next token based on what
    followed those occurrences. catches boilerplate, URLs, repeated
    paragraphs — stuff PPM-12 misses because it's capped at 12 tokens.

    single-pass, backward-looking only. same legality as n-gram cache.
    """

    def __init__(self, min_match=4, max_match=256, num_buckets=1 << 22,
                 hash_window=8):
        self.min_match = min_match
        self.max_match = max_match
        self.num_buckets = num_buckets
        self.mask = num_buckets - 1
        self.hash_window = hash_window

        # hash table: bucket -> list of positions where this hash appeared
        self.table = [[] for _ in range(num_buckets)]
        self.history = []
        self.pos = 0

        self._base = np.uint64(131)

    def _rolling_hash(self, tokens, start, length):
        h = np.uint64(0)
        for i in range(length):
            h = h * self._base + np.uint64(tokens[start + i])
        return int(h) & self.mask

    def update(self, token):
        self.history.append(token)
        if self.pos >= self.hash_window:
            h = self._rolling_hash(self.history, self.pos - self.hash_window + 1,
                                   self.hash_window)
            bucket = self.table[h]
            bucket.append(self.pos)
            if len(bucket) > 32:
                self.table[h] = bucket[-32:]
        self.pos += 1

    def predict(self, context_tokens):
        """find longest match of context_tokens[-k:] in history.
        returns (lz_prob_for_next_token, match_length, confidence) or None."""
        if len(context_tokens) < self.hash_window:
            return None

        ctx = context_tokens
        h = self._rolling_hash(ctx, len(ctx) - self.hash_window, self.hash_window)
        candidates = self.table[h]

        if not candidates:
            return None

        best_len = 0
        continuations = Counter()

        for cand_end in candidates:
            cand_start = cand_end - self.hash_window + 1
            if cand_start < 0:
                continue

            # extend match backward and forward from hash window
            match_len = self.hash_window
            ctx_pos = len(ctx) - self.hash_window
            hist_pos = cand_start

            # verify hash window actually matches
            ok = True
            for k in range(self.hash_window):
                if ctx[ctx_pos + k] != self.history[hist_pos + k]:
                    ok = False
                    break
            if not ok:
                continue

            # extend backward
            bi, bj = ctx_pos - 1, hist_pos - 1
            while bi >= 0 and bj >= 0 and ctx[bi] == self.history[bj]:
                match_len += 1
                bi -= 1
                bj -= 1
                if match_len >= self.max_match:
                    break

            if match_len >= self.min_match and cand_end + 1 < len(self.history):
                next_tok = self.history[cand_end + 1]
                continuations[next_tok] += 1
                best_len = max(best_len, match_len)

        if not continuations or best_len < self.min_match:
            return None

        total = sum(continuations.values())
        confidence = min(1.0, best_len / 50.0)

        return continuations, best_len, confidence

    def predict_token_prob(self, context_tokens, target_token):
        """P(target_token | context) from LZ copy model.
        returns (probability, confidence) or (0.0, 0.0) if no match."""
        result = self.predict(context_tokens)
        if result is None:
            return 0.0, 0.0

        continuations, match_len, confidence = result
        total = sum(continuations.values())
        p = continuations.get(target_token, 0) / total
        return p, confidence

    def update_batch(self, tokens, start, end):
        for i in range(start, end):
            self.update(tokens[i])


if __name__ == "__main__":
    # test on repetitive token sequence
    np.random.seed(42)

    # simulate: a pattern that repeats with slight variation
    pattern = list(range(50))  # 50-token pattern
    tokens = []
    for _ in range(100):
        tokens.extend(pattern)
        # small variation
        tokens.extend([np.random.randint(0, 1024) for _ in range(5)])

    lz = LZCopyModel(min_match=4, hash_window=8)

    hits, misses, total = 0, 0, 0
    for i in range(100, len(tokens)):
        ctx = tokens[:i]
        target = tokens[i]
        p, conf = lz.predict_token_prob(ctx, target)
        if conf > 0:
            hits += 1
            if p > 0:
                total += 1
        else:
            misses += 1
        lz.update(tokens[i])

    print(f"tokens: {len(tokens)}")
    print(f"hits: {hits} ({hits/(hits+misses)*100:.1f}%)")
    print(f"correct predictions: {total}")
    print(f"miss: {misses}")
