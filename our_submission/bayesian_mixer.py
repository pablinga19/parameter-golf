import numpy as np


class BayesianMixer:
    """online bayesian mixture of neural + ngram predictions.

    instead of fixed alpha, maintains posterior weight over the two
    models and updates after each scored token. total regret over
    the entire eval is bounded by ln(2) = 0.693 nats regardless
    of sequence length. converges in ~100-500 tokens.
    """

    def __init__(self, n_models=2, prior=None):
        if prior is None:
            self.log_w = np.zeros(n_models)
        else:
            p = np.array(prior, dtype=np.float64)
            assert len(p) == n_models and np.all(p > 0)
            self.log_w = np.log(p / p.sum())
        self._init_log_w = self.log_w.copy()
        self.n = n_models

    def get_weights(self):
        # numerically stable softmax
        lw = self.log_w - self.log_w.max()
        w = np.exp(lw)
        return w / w.sum()

    def mix(self, probs):
        """mix model probabilities. probs shape: (n_models,) or (n_models, n_tokens)"""
        w = self.get_weights()
        if probs.ndim == 1:
            return np.dot(w, probs)
        return (w[:, None] * probs).sum(axis=0)

    def update(self, probs_of_true_token):
        self.log_w += np.log(np.clip(probs_of_true_token, 1e-20, 1.0))
        self.log_w -= self.log_w.max()  # renormalize to prevent underflow

    def update_batch(self, probs_matrix):
        self.log_w += np.log(np.clip(probs_matrix, 1e-20, 1.0)).sum(axis=1)
        self.log_w -= self.log_w.max()

    def reset(self):
        self.log_w[:] = self._init_log_w.copy()


def mix_bayesian(p_neural, p_ngram, mixer):
    """mix neural and ngram predictions using bayesian weights.
    returns mixed probabilities (not NLL)."""
    w = mixer.get_weights()
    return w[0] * p_neural + w[1] * p_ngram


def mix_and_update(p_neural, p_ngram, true_p_neural, true_p_ngram, mixer):
    """mix predictions then update weights with true token probs.

    p_neural/p_ngram: predictions for current token (before scoring)
    true_p_neural/true_p_ngram: probability each model assigned to the
        correct token (after reveal, used for weight update)

    returns mixed probability for current token.
    """
    p_mixed = mix_bayesian(p_neural, p_ngram, mixer)
    mixer.update(np.array([true_p_neural, true_p_ngram]))
    return p_mixed
