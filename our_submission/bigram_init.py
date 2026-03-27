import numpy as np
import torch


def build_bigram_matrix(token_files, vocab_size=1024):
    """count bigram transitions from training corpus.
    returns (transition_probs, raw_counts, unigram_counts)."""
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    unigram = np.zeros(vocab_size, dtype=np.float64)
    for fpath in token_files:
        tokens = np.fromfile(fpath, dtype=np.uint16)
        for i in range(len(tokens)):
            if tokens[i] < vocab_size:
                unigram[tokens[i]] += 1
            if i > 0 and tokens[i-1] < vocab_size and tokens[i] < vocab_size:
                counts[tokens[i-1], tokens[i]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    return counts / row_sums, counts, unigram


def init_lm_head_bias(model, unigram_counts):
    """initialize lm_head bias to log-unigram probs from raw counts."""
    unigram = unigram_counts / max(unigram_counts.sum(), 1.0)
    log_unigram = np.log(np.clip(unigram, 1e-10, 1.0))

    # find the output bias if it exists
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        if model.lm_head.bias is not None:
            with torch.no_grad():
                model.lm_head.bias.copy_(torch.from_numpy(log_unigram).float())
            return True
    return False


def init_attention_from_bigrams(model, bigram_matrix, rank=32):
    """initialize layer-0 Q/K weights so attention approximates bigram
    co-occurrence structure. the first attention layer starts knowing which
    tokens typically follow which, instead of learning from scratch.

    based on enterprise research (Rende et al. arXiv:2410.19637):
    transformers fit statistics in order: mean -> covariance -> higher.
    skip the covariance phase by initializing with bigram SVD.
    """
    C = torch.from_numpy(bigram_matrix).float()
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)

    # get embedding matrix
    if hasattr(model, 'tok_emb'):
        E = model.tok_emb.weight.detach().float()
    else:
        return False

    d_model = E.shape[1]
    r = min(rank, d_model, U.shape[1])

    # project bigram modes into embedding space via least-squares
    # Q_init ≈ E^+ @ U[:, :r], K_init ≈ E^+ @ Vh[:r].T
    E_pinv = torch.linalg.pinv(E)  # (d_model, vocab)
    Q_proj = E_pinv @ U[:, :r]     # (d_model, r)
    K_proj = E_pinv @ Vh[:r].T     # (d_model, r)

    # scale to match expected weight magnitudes
    scale = (1.0 / d_model) ** 0.5

    # apply to first attention layer
    blocks = list(model.blocks) if hasattr(model, 'blocks') else []
    if not blocks:
        return False

    first_attn = None
    for name, mod in blocks[0].named_modules():
        if hasattr(mod, 'c_q') or hasattr(mod, 'q_proj'):
            first_attn = mod
            break

    if first_attn is None:
        return False

    with torch.no_grad():
        q_weight = getattr(first_attn, 'c_q', getattr(first_attn, 'q_proj', None))
        k_weight = getattr(first_attn, 'c_k', getattr(first_attn, 'k_proj', None))
        if q_weight is not None and hasattr(q_weight, 'weight'):
            w = q_weight.weight
            w[:r] = (Q_proj.T * scale).to(w.dtype)
        if k_weight is not None and hasattr(k_weight, 'weight'):
            w = k_weight.weight
            w[:r] = (K_proj.T * scale).to(w.dtype)

    return True


if __name__ == "__main__":
    # test with synthetic bigram matrix
    np.random.seed(42)
    vocab = 1024
    C = np.random.dirichlet(np.ones(vocab) * 0.1, size=vocab)

    unigram = C.sum(axis=0)
    unigram /= unigram.sum()
    entropy = -np.sum(unigram * np.log2(np.clip(unigram, 1e-10, 1.0)))
    print(f"unigram entropy: {entropy:.2f} bits (uniform={np.log2(vocab):.2f})")

    U, S, Vh = np.linalg.svd(C, full_matrices=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    r90 = np.searchsorted(energy, 0.9) + 1
    r99 = np.searchsorted(energy, 0.99) + 1
    print(f"SVD: 90% energy at rank {r90}, 99% at rank {r99}")
    print(f"top-32 captures {energy[31]*100:.1f}% of bigram structure")
