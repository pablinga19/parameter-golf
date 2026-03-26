# Technique Deep Dives

## rANS Entropy Coding for Weight Storage

Pipeline:
```
FP16 weights → INT6 quantization (lossy) → rANS entropy coding (lossless)
Result: ~3.5 bits/weight vs 6 bits/weight raw INT6
```

Best prior for transformer weights: Student-t with nu=3-6 degrees of freedom.
Layer-wise fitted priors minimize KL divergence.

Compression numbers:
| Method | Effective BPW | vs FP32 |
|--------|--------------|---------|
| INT6 alone | 6 bits | 5.3x |
| INT6 + rANS | ~3.5 bits | ~9.1x |
| INT4 + rANS (EntroLLM) | ~1.39 bits | ~23x |

Implementations:
- github.com/cambridge-mlg/miracle — Bayesian weight compression
- github.com/j-towns/craystack — vectorized ANS, numpy
- github.com/bits-back/bits-back — original BB-ANS
- github.com/LeanModels/DFloat11 — production lossless BF16

Memory note: full BB-ANS needs posterior params during decode (2x overhead).
Use plain rANS without bits-back trick for simplicity.


## Leech Lattice VQ (arXiv:2603.11021)

Encoding one 24-weight block:
1. Hadamard rotation (spreads outliers)
2. Normalize: w_hat = w / ||w||, store scale as FP16
3. Scale to target shell radius
4. Adoul-Barth nearest-neighbor via extended Golay code:
   - Group 24 dims into 4x6 chunks
   - Compute hexacode over GF(4)
   - Find nearest of 64 hexacode codewords
   - Construct lattice point via coset representative
5. Encode as (shell_index, Golay_class, local_symmetry) ~26 bits

Decoding: shell lookup → class lookup → signs/permutation → rescale → inverse Hadamard.
All table lookups, no large codebook, GPU-parallelizable.

At 2 BPW: 48 bits lattice + 16 bits scale = ~2.67 BPW effective.
Competition sweet spot: 3 BPW (beats INT4 quality, less space).

Closest code: github.com/avanpo/leech-decoding (C, Adoul-Barth algo)
512d model needs padding to multiple of 24.


## DCT Weight Codec

Concept: weight matrices have spatial correlation. DCT-II concentrates
energy into low-frequency coefficients. Store only significant coefficients.

```python
import scipy.fft
W_dct = scipy.fft.dctn(W, type=2)  # 2D DCT
W_compressed = quantize(W_dct, Q_matrix)  # JPEG-style quant table
# artifact: compressed coefficients + Q_matrix (256 bytes)
# decode: W = scipy.fft.idctn(dequantize(W_compressed, Q_matrix))
```

80% energy in top 20% of coefficients for smooth matrices.
Q_matrix can be hand-tuned or learned during QAT.
Nobody in the competition has tried this.


## Bits-Back ANS

How it works: exploit latent variables in generative model to get "free bits."
Net cost = negative ELBO. Near-Shannon-optimal as variational gap → 0.

For weight storage: treat INT6 weights as samples from a learned prior.
BB-ANS encodes at their actual information content.
Stacks ON TOP of quantization (lossless layer after lossy INT6).

Practical: skip full bits-back, use plain rANS with empirical INT6 distribution.
Still gets ~1.7x additional compression over raw INT6.
