"""
nanochat Transformer, adapted from nanochat/gpt.py.

Architecture (unchanged from nanochat, so its checkpoints load with strict=True):
  - rotary position embeddings, no learned positional embeddings
  - QK-norm (+1.2 sharpening), parameter-free RMSNorm
  - Group-Query Attention (n_kv_head <= n_head)
  - sliding-window attention pattern per layer ("SSSL")
  - ReLU^2 MLP, untied embedding / lm_head, logit soft-capping at 15
  - value embeddings (ResFormer), resid/x0 lambdas, smear gate, backout

What is new here is `forward_paged`: a flat-token forward that reads/writes a *paged*
KV cache (nano-vllm style) so many sequences of different lengths can be batched in
one step. `forward` is the plain dense forward used for training and as a numerical
reference in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoask.engine.paged_attention import AttnContext, paged_attention, store_kvcache


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Casts weights to the activation dtype (fp32 master weights, bf16 matmuls on GPU)."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    # x: (..., H, D); cos/sin broadcastable to (..., 1, D/2)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # paged KV cache views, assigned by the ModelRunner (same trick as nano-vllm)
        self.k_cache = self.v_cache = None

    def _qkv(self, x, ve, cos, sin):
        shp = x.shape[:-1]
        q = self.c_q(x).view(*shp, self.n_head, self.head_dim)
        k = self.c_k(x).view(*shp, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(*shp, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(*shp, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        return q, k, v

    def forward(self, x, ve, cos_sin, window):
        """Dense causal forward. x: (B, T, C)."""
        B, T, _ = x.shape
        q, k, v = self._qkv(x, ve, *cos_sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        rep = self.n_head // self.n_kv_head
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] <= i[:, None]) & ((i[:, None] - i[None, :]) <= window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, -1))

    def forward_paged(self, x, ve, cos_sin, window, ctx: AttnContext):
        """Flat-token forward against the paged KV cache. x: (N, C)."""
        q, k, v = self._qkv(x, ve, *cos_sin)
        store_kvcache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)
        y = paged_attention(q, self.k_cache, self.v_cache, ctx, window)
        return self.c_proj(y.reshape(x.size(0), -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # (L, D/2)
        self.register_buffer("sin", sin, persistent=False)

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def init_weights(self):
        """Same initialisation scheme as nanochat."""
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** -0.5
        for block in self.transformer.h:
            for lin in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                nn.init.uniform_(lin.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        n = self.config.n_layer
        for i in range(n):
            self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(n - 1, 1)
            self.x0_lambdas.data[i] = 0.20 - 0.15 * i / max(n - 1, 1)
        nn.init.zeros_(self.smear_lambda)
        nn.init.constant_(self.backout_lambda, 0.2)
        nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)

    def _precompute_rotary(self, seq_len, head_dim, base=100000):
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (ch / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
        return freqs.cos(), freqs.sin()

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_w = config.sequence_len
        short_w = -(-long_w // 4 // 128) * 128
        sizes = [long_w if pattern[i % len(pattern)] == "L" else short_w for i in range(config.n_layer)]
        sizes[-1] = long_w
        return sizes

    @property
    def device(self):
        return self.transformer.wte.weight.device

    @property
    def dtype(self):
        return self.transformer.wte.weight.dtype

    # ------------------------------------------------------------------ shared pieces
    def _embed(self, idx):
        return norm(self.transformer.wte(idx).to(self.dtype))

    def _smear(self, x, x_prev):
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[..., :24]))
        return x + gate * x_prev

    def _trunk(self, x, idx, layer_fn):
        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i].to(x.dtype) * x + self.x0_lambdas[i].to(x.dtype) * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = x + layer_fn(block.attn, norm(x), ve, self.window_sizes[i])
            x = x + block.mlp(norm(x))
            if i == backout_layer:
                x_backout = x
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        return norm(x)

    def compute_logits(self, h):
        logits = self.lm_head(h)[..., : self.config.vocab_size].float()
        return 15 * torch.tanh(logits / 15)

    # ------------------------------------------------------------------ dense forward
    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos_sin = self.cos[None, :T, None, :].to(self.dtype), self.sin[None, :T, None, :].to(self.dtype)
        x = self._embed(idx)
        if T > 1:
            x = torch.cat([x[:, :1], self._smear(x[:, 1:], x[:, :-1])], dim=1)
        h = self._trunk(x, idx, lambda attn, xn, ve, w: attn(xn, ve, cos_sin, w))
        logits = self.compute_logits(h)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    # ------------------------------------------------------------------ paged forward
    def forward_paged(self, input_ids, positions, prev_ids, ctx: AttnContext):
        """
        input_ids, positions, prev_ids: (N,) flat over all scheduled tokens of all sequences.
        prev_ids[j] is the token preceding input_ids[j] in its sequence (-1 at position 0);
        nanochat's smear gate needs it, and recomputing it from the token id means the
        cache needs no extra per-sequence state.
        Returns hidden states (N, C); call compute_logits on the rows you need.
        """
        cos_sin = self.cos[positions][:, None, :].to(self.dtype), self.sin[positions][:, None, :].to(self.dtype)
        x = self._embed(input_ids)
        has_prev = prev_ids >= 0
        if has_prev.any():
            x_prev = self._embed(prev_ids.clamp(min=0))
            x = torch.where(has_prev[:, None], self._smear(x, x_prev), x)
        return self._trunk(x, input_ids, lambda attn, xn, ve, w: attn.forward_paged(xn, ve, cos_sin, w, ctx))
