import torch
import torch.nn.functional as F
import numpy as np


def complementary_ce_loss(logits, targets, ngram_easy_mask, hard_weight=2.0):
    """cross-entropy loss that down-weights tokens the n-gram cache handles.

    during training, tokens that would be "easy" for the eval-time n-gram
    cache get lower loss weight. the neural model focuses capacity on
    what the cache can't predict — novel contexts, semantic reasoning,
    long-range dependencies.

    ngram_easy_mask: bool tensor, True for tokens that a simple bigram/trigram
    model would predict correctly (high-frequency n-gram patterns).
    these get weight 1/hard_weight. hard tokens get weight hard_weight.
    """
    per_token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none",
    )

    weights = torch.ones_like(per_token_loss)
    easy = ngram_easy_mask.view(-1)
    weights[easy] = 1.0 / hard_weight
    weights[~easy] = hard_weight

    # normalize so mean weight = 1.0 (don't change effective LR)
    weights = weights / weights.mean()
    return (per_token_loss * weights).mean()


def build_easy_mask_from_bigrams(tokens, bigram_counts, threshold=0.3):
    """mark tokens as "easy" if the bigram (prev_token, token) has been
    seen often enough that a simple cache would predict it.

    bigram_counts: (vocab, vocab) array of co-occurrence counts from training data.
    threshold: fraction of contexts where this bigram is the most common continuation.
    """
    vocab_size = bigram_counts.shape[0]
    # for each context token, what's the probability of the most common continuation?
    row_sums = bigram_counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    max_prob = bigram_counts.max(axis=1) / row_sums.ravel()

    # a token is "easy" if prev_token has a dominant continuation and
    # the current token IS that dominant continuation
    dominant = bigram_counts.argmax(axis=1)  # most common next token per context

    t = tokens.cpu().numpy() if hasattr(tokens, 'cpu') else tokens
    mask = np.zeros(len(t), dtype=bool)
    for i in range(1, len(t)):
        prev = int(t[i - 1])
        curr = int(t[i])
        if prev < vocab_size and curr < vocab_size:
            if dominant[prev] == curr and max_prob[prev] > threshold:
                mask[i] = True

    return torch.from_numpy(mask).to(tokens.device if hasattr(tokens, 'device') else 'cpu')


def build_easy_mask_from_entropy(logits, threshold=2.0):
    """mark tokens as "easy" based on model's own entropy.
    low entropy = model is already confident = cache would be too.
    no bigram table needed — uses the model's predictions directly.
    """
    with torch.no_grad():
        lp = F.log_softmax(logits.float(), dim=-1)
        entropy = -(lp.exp() * lp).sum(dim=-1)
    return entropy < threshold
