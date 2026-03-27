# Competitive Entry

Base: PR #761 (Score-First TTT + Multi-Order N-gram Backoff, 0.9581 BPB)

## Architecture
- 11L, 512d, GQA (8H/4KV), MLP 3x
- LeakyReLU(0.9)², XSA on all 11 layers
- Value Residual, Gated Attention, SmearGate
- BigramHash(4096), Partial RoPE (16/64), LN Scale
- EMA(0.997), warmdown=3000, int6 per-row + zstd-16

## Eval
- Sliding window stride=64
- Multi-order n-gram backoff (orders 2-7)
- Entropy-adaptive alpha
- Score-first TTT (4 epochs, AdamW lr=0.0001, freeze first 2 blocks)

## Run
```bash
SEED=1337 NGRAM_CACHE=1 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
