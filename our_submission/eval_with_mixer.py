"""Sliding-window evaluation with optional n-gram mixing."""

from __future__ import annotations
import math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from mixer import PPMDMixer, entropy_adaptive_alpha, logistic_mix


def eval_sliding(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                 is_boundary_token_lut, device, seq_len=1024, stride=64,
                 vocab_size=1024, mixer_mode=None, ngram_order=7,
                 ngram_min_order=2, ngram_buckets=4_194_304, ngram_min_count=2,
                 ent_base=0.05, ent_range=0.55, ent_scale=2.0, ent_thresh=4.0):

    mixer_mode = mixer_mode or os.environ.get("MIXER_MODE", "logistic")
    total_tokens = val_tokens.numel() - 1
    val_np = val_tokens.cpu().numpy()

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    scored = np.zeros(total_tokens + 1, dtype=bool)

    primes = [36313, 27191, 51647, 81929, 131071, 175447, 209591]
    assert ngram_buckets & (ngram_buckets - 1) == 0
    n_orders = ngram_order - ngram_min_order + 1
    ng_mask = np.uint64(ngram_buckets - 1)
    ng_primes = np.array(primes[:ngram_order], dtype=np.uint64)

    if mixer_mode == "ppmd":
        ppmd = PPMDMixer(ngram_order, ngram_min_order, ngram_buckets, ngram_min_count, primes)
    else:
        ctx_tables = [np.zeros(ngram_buckets, dtype=np.uint32) for _ in range(n_orders)]
        full_tables = [np.zeros(ngram_buckets, dtype=np.uint32) for _ in range(n_orders)]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    hits, total = 0, 0
    t0 = time.time()

    model.eval()
    fwd = getattr(model, 'forward_logits', None)
    if fwd is not None:
        try:
            fwd = torch.compile(fwd, dynamic=False, fullgraph=True)
        except Exception:
            pass  # compile not available, use uncompiled

    with torch.inference_mode():
        for wi, ws in enumerate(window_starts):
            wlen = min(ws + seq_len, total_tokens) - ws
            x = val_tokens[ws:ws + wlen].unsqueeze(0).to(device)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = (fwd or model)(x)
            else:
                logits = (fwd or model)(x)

            y = val_tokens[ws + 1:ws + wlen + 1].unsqueeze(0).to(device)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  y.reshape(-1), reduction="none").view(1, -1)

            s = 0
            for j in range(wlen):
                if not scored[ws + 1 + j]:
                    s = j
                    break
            else:
                continue

            seg_nll = nll[0, s:wlen].to(torch.float64)
            n_seg = seg_nll.numel()
            gj = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)

            with torch.no_grad():
                lp = F.log_softmax(logits[0, s:wlen].float(), dim=-1)
                seg_ent = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()

            seg_nll_np = seg_nll.cpu().numpy()
            seg_p = np.exp(-seg_nll_np)

            if mixer_mode == "ppmd":
                p_ng, has = ppmd.predict_blended(val_np, gj, n_seg)
                hits += has.sum()
                total += n_seg
                if has.any():
                    a = np.clip(entropy_adaptive_alpha(seg_ent[has], ent_base, ent_range, ent_scale, ent_thresh), 0, 0.95)
                    seg_p[has] = logistic_mix(seg_p[has], p_ng[has], a)
                seg_nll_np = -np.log(np.clip(seg_p, 1e-12, 1.0))
                ppmd.update_tables(val_np, int(gj[0]), int(gj[-1]) + 1)

            else:
                best = np.full(n_seg, -1.0)
                for oi in range(n_orders):
                    ctx_len = ngram_min_order + oi - 1
                    valid = gj >= ctx_len
                    if not valid.any(): continue
                    vi = np.nonzero(valid)[0]
                    ck = np.zeros(len(vi), dtype=np.uint64)
                    for k in range(ctx_len):
                        ti = np.clip(gj[vi] - 1 - k, 0, len(val_np) - 1)
                        ck ^= ng_primes[k] * val_np[ti].astype(np.uint64)
                    ck &= ng_mask
                    ti = np.clip(gj[vi], 0, len(val_np) - 1)
                    pidx = min(ctx_len, len(ng_primes) - 1)
                    fk = ck ^ (ng_primes[pidx] * val_np[ti].astype(np.uint64))
                    fk &= ng_mask
                    cc = ctx_tables[oi][ck].astype(np.float64)
                    fc = full_tables[oi][fk].astype(np.float64)
                    got = cc >= float(ngram_min_count)
                    need = got & (best[vi] < 0)
                    if need.any():
                        fi = vi[need]
                        best[fi] = np.clip(np.minimum(fc[need], cc[need]) / np.maximum(cc[need], 1.0), 0, 1)

                has = best >= 0
                hits += has.sum()
                total += n_seg
                if has.any():
                    a = np.clip(entropy_adaptive_alpha(seg_ent[has], ent_base, ent_range, ent_scale, ent_thresh), 0, 0.95)
                    if mixer_mode == "logistic":
                        seg_p[has] = logistic_mix(seg_p[has], best[has], a)
                    else:
                        seg_p[has] = (1.0 - a) * seg_p[has] + a * best[has]
                seg_nll_np = -np.log(np.clip(seg_p, 1e-12, 1.0))

                for oi in range(n_orders):
                    ctx_len = ngram_min_order + oi - 1
                    for jl in range(n_seg):
                        jg = int(gj[jl])
                        if jg < ctx_len: continue
                        ck = np.uint64(0)
                        for k in range(ctx_len):
                            ck ^= ng_primes[k] * np.uint64(val_np[jg - 1 - k])
                        ck &= ng_mask
                        pidx = min(ctx_len, len(ng_primes) - 1)
                        fk = ck ^ (ng_primes[pidx] * np.uint64(val_np[jg]))
                        fk &= ng_mask
                        ctx_tables[oi][ck] += 1
                        full_tables[oi][fk] += 1

            mixed = torch.from_numpy(seg_nll_np).to(device=device, dtype=torch.float64)
            loss_sum += mixed.sum()
            tok_count += n_seg

            tgt = val_np[gj].astype(np.int64)
            prev = val_np[np.clip(gj - 1, 0, len(val_np) - 1)].astype(np.int64)
            b = base_bytes_lut[tgt].cpu().numpy().astype(np.float64)
            b += (has_leading_space_lut[tgt].cpu().numpy() &
                  ~is_boundary_token_lut[prev].cpu().numpy()).astype(np.float64)
            byte_count += b.sum()

            for jl in range(n_seg):
                scored[int(gj[jl])] = True

            if (wi + 1) % 500 == 0:
                elapsed = time.time() - t0
                pct = (wi + 1) / len(window_starts) * 100
                bpb = (loss_sum / tok_count / math.log(2) * tok_count / byte_count).item() if byte_count > 0 else 0
                hr = hits / max(total, 1) * 100
                print(f"  [{mixer_mode}] {pct:.1f}% bpb={bpb:.4f} hits={hr:.1f}% {elapsed:.0f}s", flush=True)

    if tok_count == 0 or byte_count == 0:
        return 0.0, 0.0, {}

    vl = (loss_sum / tok_count).item()
    bpt = vl / math.log(2.0)
    tpb = tok_count.item() / byte_count.item()

    return vl, bpt * tpb, {
        "mixer": mixer_mode, "val_loss": vl, "val_bpb": bpt * tpb,
        "hit_rate": hits / max(total, 1), "tokens": int(tok_count.item()),
        "time": time.time() - t0,
    }
