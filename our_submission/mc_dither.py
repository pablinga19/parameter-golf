import math
import torch
import torch.nn.functional as F


def dithered_logits(model, x, n_draws=4, noise_scale=None, clip_range=31):
    if n_draws <= 1:
        fwd = getattr(model, 'forward_logits', model)
        return fwd(x)

    originals = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            originals[name] = p.data.clone()

    logit_sum = None
    try:
        for _ in range(n_draws):
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name not in originals:
                        continue
                    if noise_scale is not None:
                        delta = originals[name].abs().mean() * noise_scale
                    else:
                        row_max = originals[name].abs().amax(dim=-1, keepdim=True)
                        delta = row_max / clip_range
                    noise = torch.empty_like(p).uniform_(-0.5, 0.5) * delta
                    p.copy_(originals[name] + noise)

            fwd = getattr(model, 'forward_logits', model)
            with torch.inference_mode():
                logits = fwd(x)

            if logit_sum is None:
                logit_sum = logits.float()
            else:
                logit_sum += logits.float()
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in originals:
                    p.copy_(originals[name])

    return logit_sum / n_draws


def dithered_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                 is_boundary_token_lut, device, seq_len=1024,
                 n_draws=4, noise_scale=None):
    total_tokens = val_tokens.numel() - 1
    total_seqs = total_tokens // seq_len
    if total_seqs == 0:
        raise ValueError(f"need at least {seq_len + 1} tokens")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    for seq_i in range(total_seqs):
        start = seq_i * seq_len
        x = val_tokens[start:start + seq_len].unsqueeze(0).to(device)
        y = val_tokens[start + 1:start + seq_len + 1].unsqueeze(0).to(device)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = dithered_logits(model, x, n_draws=n_draws,
                                         noise_scale=noise_scale)
        else:
            logits = dithered_logits(model, x, n_draws=n_draws,
                                     noise_scale=noise_scale)

        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              y.reshape(-1), reduction="none")
        loss_sum += nll.to(torch.float64).sum()
        tok_count += float(y.numel())

        prev = x.reshape(-1)
        tgt = y.reshape(-1)
        tb = base_bytes_lut[tgt].to(torch.float64)
        tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
        byte_count += tb.sum()

    val_loss = (loss_sum / tok_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = tok_count.item() / byte_count.item()
    return val_loss, bpt * tpb
