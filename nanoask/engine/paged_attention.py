"""
Paged attention in pure PyTorch.

nano-vllm uses a Triton kernel to scatter K/V into the paged cache and flash-attn's
`flash_attn_varlen_func` / `flash_attn_with_kvcache(block_table=...)` to read it back.
Those need CUDA, so this module implements the same contract with plain tensor ops:
it runs on CPU, and on GPU it is slower than flash-attn but numerically equivalent.

KV cache layout per layer:  (num_blocks, block_size, n_kv_head, head_dim)
A token's "slot" is block_id * block_size + offset_in_block.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttnContext:
    is_prefill: bool
    slot_mapping: torch.Tensor      # (N,) slot for every new token (-1 = don't store)
    context_lens: torch.Tensor      # (B,) total tokens visible to each seq after this step
    block_tables: torch.Tensor      # (B, max_blocks) block ids, -1 padded
    cu_seqlens_q: torch.Tensor      # (B+1,) prefix sums of new tokens per seq
    block_size: int
    max_query_len: int = 1


def store_kvcache(k, v, k_cache, v_cache, slot_mapping):
    """Scatter new K/V rows into their cache slots. k, v: (N, Hkv, D)."""
    if k_cache is None:
        return
    keep = slot_mapping >= 0
    slots = slot_mapping[keep].long()
    k_cache.view(-1, *k_cache.shape[2:])[slots] = k[keep].to(k_cache.dtype)
    v_cache.view(-1, *v_cache.shape[2:])[slots] = v[keep].to(v_cache.dtype)


def _gather(cache, block_tables, n_tokens):
    """(B, max_blocks) block ids -> (B, max_blocks*block_size, Hkv, D), truncated to n_tokens."""
    bt = block_tables.clamp(min=0).long()
    g = cache[bt]  # (B, nb, bs, H, D)
    B, nb, bs, H, D = g.shape
    return g.reshape(B, nb * bs, H, D)[:, :n_tokens]


def paged_attention(q, k_cache, v_cache, ctx: AttnContext, window: int):
    """
    q: (N, H, D) queries for all new tokens, grouped by sequence via ctx.cu_seqlens_q.
    Each sequence i has `n_i` new tokens at absolute positions [L_i - n_i, L_i) where
    L_i = context_lens[i]; it attends causally to cached keys [0, L_i), restricted to a
    left window of `window` tokens (nanochat sliding-window semantics).
    Everything is padded to (B, max_q, ...) and done in one SDPA call: that is the batching.
    """
    N, H, D = q.shape
    B = ctx.context_lens.numel()
    Hkv = k_cache.shape[2]
    max_k = int(ctx.context_lens.max())
    keys = _gather(k_cache, ctx.block_tables, max_k).to(q.dtype)   # (B, Lk, Hkv, D)
    vals = _gather(v_cache, ctx.block_tables, max_k).to(q.dtype)

    q_lens = ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]           # (B,)
    max_q = ctx.max_query_len
    dev = q.device
    # scatter flat queries into a padded (B, max_q, H, D) tensor
    seq_idx = torch.repeat_interleave(torch.arange(B, device=dev), q_lens)
    pos_in_q = torch.arange(N, device=dev) - ctx.cu_seqlens_q[:-1].long()[seq_idx]
    qp = q.new_zeros(B, max_q, H, D)
    qp[seq_idx, pos_in_q] = q

    # absolute query positions and key positions -> mask (B, max_q, Lk)
    q_pos = (ctx.context_lens - q_lens)[:, None] + torch.arange(max_q, device=dev)[None, :]
    k_pos = torch.arange(max_k, device=dev)
    rel = q_pos[:, :, None] - k_pos[None, None, :]
    mask = (rel >= 0) & (rel <= window) & (k_pos[None, None, :] < ctx.context_lens[:, None, None])

    rep = H // Hkv
    if rep > 1:
        keys, vals = keys.repeat_interleave(rep, dim=2), vals.repeat_interleave(rep, dim=2)
    out = F.scaled_dot_product_attention(
        qp.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2), attn_mask=mask[:, None]
    )  # (B, H, max_q, D)
    return out.transpose(1, 2)[seq_idx, pos_in_q]                   # back to (N, H, D)
