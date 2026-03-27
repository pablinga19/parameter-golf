#!/bin/bash
set -e
echo "=== COMPETITIVE ENTRY DEPLOY ==="
echo "base: PR #761 (0.9581 BPB, 3-seed validated)"
echo "start: $(date)"

cd /workspace
git clone https://github.com/openai/parameter-golf.git pg 2>&1 | tail -1
cd pg
pip install sentencepiece flash-attn zstandard huggingface_hub -q 2>&1 | tail -1
python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -3

# use PR #761 train_gpt.py (fetched from our fork)
git clone -b ppmd-submission https://github.com/pablinga19/parameter-golf.git /workspace/ours 2>&1 | tail -1
cp /workspace/ours/competitive_entry/train_gpt.py .

echo "--- TRAINING (seed 1337) ---"
SEED=1337 NGRAM_CACHE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tail -15

echo "--- TRAINING (seed 42) ---"
SEED=42 NGRAM_CACHE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tail -15

echo "--- TRAINING (seed 7) ---"
SEED=7 NGRAM_CACHE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tail -15

echo "=== DONE ==="
echo "end: $(date)"
