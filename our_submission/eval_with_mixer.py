"""
Drop-in sliding window eval with pluggable mixer.

Replaces the n-gram eval section in any train_gpt.py.
Usage: import and call eval_sliding_with_mixer() instead of the default eval.

Supports 3 mixing modes via MIXER_MODE env var:
  - "linear"   (current SOTA, baseline comparison)
  - "logistic"  (PAQ-style log-odds mixing)
  - "ppmd"      (PPM-D blended order mixing + logistic)
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mixer import (
    CTWMixer,
    PPMDMixer,
    entropy_adaptive_alpha,
    logistic_mix,
    mix_predictions,
)


def eval_sliding_with_mixer(
    model: torch.nn.Module,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
    seq_len: int = 1024,
    stride: int = 64,
    vocab_size: int = 1024,
    # Mixer config
    mixer_mode: str | None = None,
    ngram_order: int = 7,
    ngram_min_order: int = 2,
    ngram_buckets: int = 4_194_304,
    ngram_min_count: int = 2,
    ngram_ent_base: float = 0.05,
    ngram_ent_range: float = 0.55,
    ngram_ent_scale: float = 2.0,
    ngram_ent_thresh: float = 4.0,
) -> tuple[float, float, dict]:
    """Sliding window eval with pluggable probability mixer.

    Returns:
        val_loss: mean NLL (nats)
        val_bpb: bits per byte
        stats: dict with timing, hit rates, etc.
    """
    if mixer_mode is None:
        mixer_mode = os.environ.get("MIXER_MODE", "logistic")

    total_tokens = val_tokens.numel() - 1
    val_np = val_tokens.cpu().numpy()

    # Build window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    # Track which tokens have been scored (for score-first protocol)
    scored = np.zeros(total_tokens + 1, dtype=bool)

    # Initialize mixer
    primes = [36313, 27191, 51647, 81929, 131071, 175447, 209591]

    if mixer_mode in ("linear", "logistic"):
        # Use simple hash tables (same structure as current SOTA)
        n_orders = ngram_order - ngram_min_order + 1
        ctx_tables = [np.zeros(ngram_buckets, dtype=np.uint32) for _ in range(n_orders)]
        full_tables = [np.zeros(ngram_buckets, dtype=np.uint32) for _ in range(n_orders)]
        ng_mask = np.uint64(ngram_buckets - 1)
        ng_primes = np.array(primes[:ngram_order], dtype=np.uint64)
    elif mixer_mode == "ppmd":
        ppmd = PPMDMixer(
            max_order=ngram_order,
            min_order=ngram_min_order,
            num_buckets=ngram_buckets,
            min_count=ngram_min_count,
            primes=primes,
        )

    # Accumulators
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    ngram_hits = 0
    ngram_total = 0
    t0 = time.time()

    model.eval()
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)

    with torch.inference_mode():
        for wi, ws in enumerate(window_starts):
            wlen = min(ws + seq_len, total_tokens) - ws
            x_batch = val_tokens[ws:ws + wlen].unsqueeze(0).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)  # (1, wlen, vocab)

            y_batch = val_tokens[ws + 1:ws + wlen + 1].unsqueeze(0).to(device)
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1),
                reduction="none",
            ).view(1, -1)

            # Determine which tokens in this window are newly scored
            s = 0
            for j in range(wlen):
                if not scored[ws + 1 + j]:
                    s = j
                    break
            else:
                continue  # all tokens already scored

            # Score the new tokens
            scored_nll = nll[0, s:wlen].to(torch.float64)
            n_seg = scored_nll.numel()
            global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)

            # Compute model entropy for adaptive alpha
            with torch.no_grad():
                lp = F.log_softmax(logits[0, s:wlen].float(), dim=-1)
                seg_ent = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()

            # Get model probability for correct token
            seg_nll_np = scored_nll.cpu().numpy()
            seg_model_p = np.exp(-seg_nll_np)

            # --- MIXER APPLICATION ---
            if mixer_mode in ("linear", "logistic"):
                # Same hash-based n-gram as SOTA, but with configurable mixing
                best_p_ng = np.full(n_seg, -1.0)

                for oi in range(n_orders):
                    order = ngram_min_order + oi
                    ctx_len = order - 1
                    valid = global_j >= ctx_len
                    if not valid.any():
                        continue

                    v_idx = np.nonzero(valid)[0]

                    # Hash PRECEDING tokens (score-first: never include target)
                    ctx_keys = np.zeros(len(v_idx), dtype=np.uint64)
                    for k in range(ctx_len):
                        tok_idx = np.clip(global_j[v_idx] - 1 - k, 0, len(val_np) - 1)
                        ctx_keys ^= ng_primes[k] * val_np[tok_idx].astype(np.uint64)
                    ctx_keys &= ng_mask

                    # Full key = context + target
                    target_idx = np.clip(global_j[v_idx], 0, len(val_np) - 1)
                    prime_idx = min(ctx_len, len(ng_primes) - 1)
                    full_keys = ctx_keys ^ (ng_primes[prime_idx] * val_np[target_idx].astype(np.uint64))
                    full_keys &= ng_mask

                    ctx_counts = ctx_tables[oi][ctx_keys].astype(np.float64)
                    full_counts = full_tables[oi][full_keys].astype(np.float64)

                    has_match = ctx_counts >= float(ngram_min_count)
                    needs_fill = has_match & (best_p_ng[v_idx] < 0)
                    if needs_fill.any():
                        fill_idx = v_idx[needs_fill]
                        p = np.minimum(full_counts[needs_fill], ctx_counts[needs_fill]) / np.maximum(ctx_counts[needs_fill], 1.0)
                        best_p_ng[fill_idx] = np.clip(p, 0.0, 1.0)

                has_match_mask = best_p_ng >= 0
                ngram_hits += has_match_mask.sum()
                ngram_total += n_seg

                if has_match_mask.any():
                    alpha = np.clip(
                        entropy_adaptive_alpha(seg_ent[has_match_mask], ngram_ent_base, ngram_ent_range, ngram_ent_scale, ngram_ent_thresh),
                        0.0, 0.95,
                    )

                    if mixer_mode == "linear":
                        seg_model_p[has_match_mask] = (1.0 - alpha) * seg_model_p[has_match_mask] + alpha * best_p_ng[has_match_mask]
                    elif mixer_mode == "logistic":
                        seg_model_p[has_match_mask] = logistic_mix(
                            seg_model_p[has_match_mask],
                            best_p_ng[has_match_mask],
                            alpha,
                        )

                seg_nll_np = -np.log(np.clip(seg_model_p, 1e-12, 1.0))

                # Update tables AFTER scoring (score-first protocol)
                for oi in range(n_orders):
                    order = ngram_min_order + oi
                    ctx_len = order - 1
                    for j_local in range(n_seg):
                        j_global = int(global_j[j_local])
                        if j_global < ctx_len:
                            continue
                        ck = np.uint64(0)
                        for k in range(ctx_len):
                            ck ^= ng_primes[k] * np.uint64(val_np[j_global - 1 - k])
                        ck &= ng_mask
                        p_idx = min(ctx_len, len(ng_primes) - 1)
                        fk = ck ^ (ng_primes[p_idx] * np.uint64(val_np[j_global]))
                        fk &= ng_mask
                        ctx_tables[oi][ck] += 1
                        full_tables[oi][fk] += 1

            elif mixer_mode == "ppmd":
                p_blend, has_match_mask = ppmd.predict_blended(val_np, global_j, n_seg)
                ngram_hits += has_match_mask.sum()
                ngram_total += n_seg

                if has_match_mask.any():
                    alpha = np.clip(
                        entropy_adaptive_alpha(seg_ent[has_match_mask], ngram_ent_base, ngram_ent_range, ngram_ent_scale, ngram_ent_thresh),
                        0.0, 0.95,
                    )
                    # PPM-D always uses logistic mixing
                    seg_model_p[has_match_mask] = logistic_mix(
                        seg_model_p[has_match_mask],
                        p_blend[has_match_mask],
                        alpha,
                    )

                seg_nll_np = -np.log(np.clip(seg_model_p, 1e-12, 1.0))
                ppmd.update_tables(val_np, int(global_j[0]), int(global_j[-1]) + 1)

            # Accumulate loss
            scored_nll_mixed = torch.from_numpy(seg_nll_np).to(device=device, dtype=torch.float64)
            loss_sum += scored_nll_mixed.sum()
            token_count += n_seg

            # Byte counting
            for j_local in range(n_seg):
                j_global = int(global_j[j_local])
                tgt_id = int(val_np[j_global])
                prev_id = int(val_np[j_global - 1]) if j_global > 0 else 0
                b = base_bytes_lut[tgt_id].item()
                if has_leading_space_lut[tgt_id].item() and not is_boundary_token_lut[prev_id].item():
                    b += 1
                byte_count += b

            # Mark scored
            for j_local in range(n_seg):
                scored[int(global_j[j_local])] = True

            # Progress
            if (wi + 1) % 500 == 0:
                elapsed = time.time() - t0
                pct = (wi + 1) / len(window_starts) * 100
                current_bpb = (loss_sum / token_count / math.log(2) * token_count / byte_count).item() if byte_count > 0 else 0
                hit_rate = ngram_hits / max(ngram_total, 1) * 100
                print(f"  [{mixer_mode}] {pct:.1f}% | bpb={current_bpb:.4f} | hits={hit_rate:.1f}% | {elapsed:.0f}s", flush=True)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    val_bpb = bits_per_token * tokens_per_byte
    elapsed = time.time() - t0

    stats = {
        "mixer_mode": mixer_mode,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "ngram_hit_rate": ngram_hits / max(ngram_total, 1),
        "tokens_scored": int(token_count.item()),
        "eval_time_s": elapsed,
    }

    return val_loss, val_bpb, stats
