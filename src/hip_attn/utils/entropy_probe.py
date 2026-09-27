"""Causal entropy probes for an unpadded, contiguous Q/K sequence."""
import torch


@torch.no_grad()
def causal_probe_entropy(q, k, probe_q=128, probe_k=1024):
    """Return per-probed-query/head entropy; q must already be softmax-scaled.

    Queries are aligned to the last Tq positions of the key sequence, as in
    causal prefill and contiguous KV-cache decoding. The caller aggregates this
    signal into one budget for the complete layer forward.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("Expected [batch, sequence, heads, head_dim] tensors")
    n, tq, hq, d = q.shape
    nk, tk, hk, dk = k.shape
    if n != nk or d != dk or tq > tk or hk <= 0 or hq % hk:
        raise ValueError("Incompatible Q/K shapes or GQA head ratio")
    if tq == 0 or tk == 0 or probe_q <= 0 or probe_k <= 0:
        raise ValueError("Sequences and probe sizes must be positive")
    Q, P = min(int(probe_q), tq), min(int(probe_k), tk)
    # With a short key probe, exclude earlier queries having no visible key.
    Q = min(Q, P)
    curr_q = q[:, -Q:].float()
    curr_k = k[:, -P:].float().repeat_interleave(hq // hk, dim=2)
    logits = torch.einsum("nqhd,nphd->nqhp", curr_q, curr_k)
    query_positions = torch.arange(tk - Q, tk, device=q.device)
    key_positions = torch.arange(tk - P, tk, device=q.device)
    visible = key_positions[None, :] <= query_positions[:, None]
    logits.masked_fill_(~visible[None, :, None, :], float("-inf"))
    probs = logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(-1)
    visible_counts = visible.sum(-1)
    denominator = visible_counts.float().clamp_min(2).log()[None, :, None]
    return (entropy / denominator).clamp(0., 1.)


def estimate_entropy_norm(q, k, probe_q=128, probe_k=1024):
    return float(causal_probe_entropy(q, k, probe_q, probe_k).mean().item())
