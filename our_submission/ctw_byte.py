import math
import numpy as np


class KTByte:
    """krichevsky-trofimov estimator for 256-symbol byte alphabet.
    128 total pseudocounts vs 25K+ at token level — the whole point."""
    __slots__ = ('counts', 'total')

    def __init__(self):
        self.counts = np.zeros(256, dtype=np.float64)
        self.total = 0.0

    def predict(self, byte_val):
        return (self.counts[byte_val] + 0.5) / (self.total + 128.0)

    def update(self, byte_val):
        self.counts[byte_val] += 1.0
        self.total += 1.0


class CTWNode:
    __slots__ = ('kt', 'children')

    def __init__(self):
        self.kt = KTByte()
        self.children = {}


class ByteCTW:
    """byte-level context tree weighting.

    operates on raw UTF-8 bytes (alphabet=256). each node maintains
    a KT estimator. the tree mixes predictions bottom-up with exact
    1/2 bayesian weighting per the original Willems 1995 paper.

    for token-level predictions, use token_prob() which decomposes
    a BPE token into its UTF-8 bytes and computes the joint probability.
    """

    def __init__(self, max_depth=12):
        self.max_depth = max_depth
        self.root = CTWNode()
        self._byte_ctx = []  # rolling byte context

    def predict_byte(self, byte_val):
        """predict P(byte_val | context) using CTW mixture.

        walks the tree bottom-up, computing:
          at leaf:     P_w = P_kt
          at internal: P_w = 0.5 * P_kt + 0.5 * P_children
        """
        ctx = self._byte_ctx

        # collect nodes from root (depth=0) to max_depth
        nodes = []
        node = self.root
        nodes.append(node)
        for d in range(min(self.max_depth, len(ctx))):
            b = ctx[-(d + 1)]
            if b not in node.children:
                break
            node = node.children[b]
            nodes.append(node)

        # bottom-up: compute P_w at each depth
        # deepest node: P_w = P_kt
        p_w = nodes[-1].kt.predict(byte_val)

        # propagate up
        for i in range(len(nodes) - 2, -1, -1):
            p_kt = nodes[i].kt.predict(byte_val)
            p_w = 0.5 * p_kt + 0.5 * p_w  # exact CTW 1/2 mixing

        return max(p_w, 1e-20)

    def update_byte(self, byte_val):
        """observe byte_val, update all nodes on the context path."""
        ctx = self._byte_ctx

        # update KT estimators at each depth
        self.root.kt.update(byte_val)
        node = self.root
        for d in range(min(self.max_depth, len(ctx))):
            b = ctx[-(d + 1)]
            if b not in node.children:
                node.children[b] = CTWNode()
            node = node.children[b]
            node.kt.update(byte_val)

        # advance context
        self._byte_ctx.append(byte_val)
        if len(self._byte_ctx) > self.max_depth + 1:
            self._byte_ctx = self._byte_ctx[-(self.max_depth + 1):]

    def token_prob(self, token_bytes):
        """P(token | byte context) = product of P(byte_i | ctx, bytes_before_i).

        temporarily advances byte context through the token without
        updating counts. restores context state after scoring.
        """
        saved_ctx = self._byte_ctx[:]
        prob = 1.0
        for b in token_bytes:
            prob *= self.predict_byte(b)
            # advance context for next byte in token (no count update)
            self._byte_ctx.append(b)
            if len(self._byte_ctx) > self.max_depth + 1:
                self._byte_ctx = self._byte_ctx[-(self.max_depth + 1):]
        self._byte_ctx = saved_ctx  # restore — counts updated separately
        return max(prob, 1e-300)

    def update_token(self, token_bytes):
        """update CTW after scoring a token. call AFTER token_prob."""
        for b in token_bytes:
            self.update_byte(b)

    def node_count(self):
        """count total nodes in tree (for memory profiling)."""
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count


class BayesianCTWMixer:
    """combines neural model + byte-level CTW via bayesian multiplicative update.

    maintains log-weights for each model. after each scored token,
    updates weights proportional to each model's probability for the
    true token. total regret bounded by ln(2) = 0.693 nats.
    """

    def __init__(self, ctw_depth=12):
        self.ctw = ByteCTW(max_depth=ctw_depth)
        self.log_w = np.array([0.0, 0.0])  # [neural, ctw]

    def _weights(self):
        lw = self.log_w - self.log_w.max()
        w = np.exp(lw)
        return w / w.sum()

    def predict(self, p_neural, token_bytes):
        """get mixed prediction for current token.

        p_neural: neural model's probability for the correct token
        token_bytes: UTF-8 bytes of the token being predicted
        """
        p_ctw = self.ctw.token_prob(token_bytes)
        w = self._weights()
        return w[0] * p_neural + w[1] * p_ctw

    def update(self, p_neural, token_bytes):
        """update weights and CTW after observing the true token.
        call AFTER predict, once per scored token."""
        p_ctw = self.ctw.token_prob(token_bytes)

        self.log_w[0] += math.log(max(p_neural, 1e-20))
        self.log_w[1] += math.log(max(p_ctw, 1e-20))
        self.log_w -= self.log_w.max()  # renormalize

        self.ctw.update_token(token_bytes)


if __name__ == "__main__":
    # test on english-like byte sequences
    text = b"the cat sat on the mat. the cat sat on the mat. the dog sat on the rug. "
    text = text * 50  # repeat for statistics

    ctw = ByteCTW(max_depth=8)
    total_nll = 0.0
    n = 0

    for i, b in enumerate(text):
        p = ctw.predict_byte(b)
        total_nll += -math.log2(max(p, 1e-20))
        ctw.update_byte(b)
        n += 1

    bpb = total_nll / len(text)
    print(f"byte-level CTW on repeated english: {bpb:.3f} BPB ({n} bytes, {ctw.node_count()} nodes)")

    # compare to uniform baseline
    uniform_bpb = math.log2(256)
    print(f"uniform baseline: {uniform_bpb:.3f} BPB")
    print(f"CTW improvement: {uniform_bpb - bpb:.3f} BPB ({(uniform_bpb - bpb)/uniform_bpb*100:.1f}%)")

    # test token bridge
    ctw2 = ByteCTW(max_depth=8)
    for b in b"the cat sat on the ":
        ctw2.update_byte(b)

    p_the = ctw2.token_prob(list(b"mat"))
    p_dog = ctw2.token_prob(list(b"dog"))
    print(f"\nafter 'the cat sat on the ': P('mat')={p_the:.6f} P('dog')={p_dog:.6f}")
