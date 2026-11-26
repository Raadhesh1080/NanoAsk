"""
ModelRunner: owns the model + the paged KV-cache pool and executes one scheduled batch.
Adapted from nano-vllm/engine/model_runner.py (single device; no tensor parallelism and no
CUDA graphs, so it also runs on CPU).

Prefill and decode share one code path: every scheduled sequence contributes
`num_scheduled_tokens` new tokens starting at `num_cached_tokens` (decode is just the case
num_scheduled_tokens == 1). All of them are flattened into a single (N,) batch.
"""
from __future__ import annotations

import torch

from nanoask.engine.paged_attention import AttnContext
from nanoask.engine.sequence import Sequence
from nanoask.model.gpt import GPT


def kv_bytes_per_block(model: GPT, block_size: int, dtype: torch.dtype) -> int:
    c = model.config
    head_dim = c.n_embd // c.n_head
    return 2 * c.n_layer * block_size * c.n_kv_head * head_dim * torch.tensor([], dtype=dtype).element_size()


class ModelRunner:
    def __init__(self, model: GPT, block_size: int, num_kvcache_blocks: int | None = None,
                 kv_cache_mb: float = 256, gpu_memory_utilization: float = 0.9, seed: int = 0,
                 max_num_batched_tokens: int = 2048):
        self.model = model.eval()
        self.device = model.device
        self.block_size = block_size
        self.kv_dtype = model.dtype
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.max_num_batched_tokens = max_num_batched_tokens
        # A previous engine on the same model left views of its KV cache on the attention modules.
        # Drop them first, otherwise that cache stays allocated and this engine sees no free memory.
        for block in self.model.transformer.h:
            block.attn.k_cache = block.attn.v_cache = None
        if self.device.type == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        if num_kvcache_blocks is None:
            num_kvcache_blocks = self._blocks_from_budget(kv_cache_mb, gpu_memory_utilization)
        assert num_kvcache_blocks > 0
        self.num_kvcache_blocks = num_kvcache_blocks
        self._allocate_kv_cache()

    def _blocks_from_budget(self, kv_cache_mb, gpu_memory_utilization):
        per_block = kv_bytes_per_block(self.model, self.block_size, self.kv_dtype)
        if self.device.type == "cuda":
            # nano-vllm style: run the largest batch the scheduler can send *before* allocating the
            # cache, measure the activation peak, and give the KV cache what is left under the cap.
            # (Without this the cache took 90% of the GPU and the first big prefill ran out of memory.)
            act_peak = self._warmup_peak_bytes()
            free, total = torch.cuda.mem_get_info(self.device)
            headroom = 1.5 * act_peak + 512 * 2**20   # paged prefill pads queries/keys; keep slack
            budget = total * gpu_memory_utilization - (total - free) - headroom
            if budget < per_block:
                raise RuntimeError(f"not enough GPU memory for a KV cache: {free / 2**30:.1f} GiB free, "
                                   f"activations need ~{headroom / 2**30:.1f} GiB. Lower --max-num-batched-tokens.")
        else:
            budget = kv_cache_mb * 2**20
        return int(budget // per_block)

    @torch.inference_mode()
    def _warmup_peak_bytes(self) -> int:
        """Peak extra memory of a dense forward over max_num_batched_tokens tokens (as nano-vllm's warmup)."""
        seq_len = min(self.max_num_batched_tokens, self.model.config.sequence_len)
        num_seqs = max(1, self.max_num_batched_tokens // seq_len)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        before = torch.cuda.memory_allocated(self.device)
        ids = torch.zeros(num_seqs, seq_len, dtype=torch.long, device=self.device)
        self.model(ids)
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device) - before
        torch.cuda.empty_cache()
        return peak

    def _allocate_kv_cache(self):
        c = self.model.config
        head_dim = c.n_embd // c.n_head
        self.kv_cache = torch.zeros(2, c.n_layer, self.num_kvcache_blocks, self.block_size,
                                    c.n_kv_head, head_dim, dtype=self.kv_dtype, device=self.device)
        for i, block in enumerate(self.model.transformer.h):
            block.attn.k_cache = self.kv_cache[0, i]
            block.attn.v_cache = self.kv_cache[1, i]

    @property
    def kv_cache_bytes(self):
        return self.kv_cache.numel() * self.kv_cache.element_size()

    # ------------------------------------------------------------------ batch construction
    def prepare(self, seqs: list[Sequence], is_prefill: bool):
        input_ids, positions, prev_ids, slots = [], [], [], []
        cu_q, ctx_lens = [0], []
        bs = self.block_size
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            prev_ids.extend(seq[p - 1] if p > 0 else -1 for p in range(start, end))
            slots.extend(seq.block_table[p // bs] * bs + p % bs for p in range(start, end))
            cu_q.append(cu_q[-1] + (end - start))
            ctx_lens.append(end)
        max_blocks = max(len(s.block_table) for s in seqs)
        tables = [s.block_table + [-1] * (max_blocks - len(s.block_table)) for s in seqs]
        dev = self.device
        t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=dev)
        ctx = AttnContext(
            is_prefill=is_prefill,
            slot_mapping=t(slots),
            context_lens=t(ctx_lens),
            block_tables=t(tables),
            cu_seqlens_q=t(cu_q),
            block_size=bs,
            max_query_len=max(seq.num_scheduled_tokens for seq in seqs),
        )
        return t(input_ids), t(positions), t(prev_ids), ctx

    # ------------------------------------------------------------------ sampling
    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        out = []
        for row, seq in zip(logits, seqs):
            sp = seq.sampling_params
            if sp.temperature <= 0:
                out.append(int(row.argmax()))
                continue
            row = row / sp.temperature
            if sp.top_k:
                v, _ = torch.topk(row, min(sp.top_k, row.numel()))
                row = row.masked_fill(row < v[-1], float("-inf"))
            probs = torch.softmax(row, dim=-1)
            out.append(int(torch.multinomial(probs, 1, generator=self.generator)))
        return out

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions, prev_ids, ctx = self.prepare(seqs, is_prefill)
        hidden = self.model.forward_paged(input_ids, positions, prev_ids, ctx)
        last = ctx.cu_seqlens_q[1:] - 1   # only the last new token of each seq is sampled
        logits = self.model.compute_logits(hidden[last])
        return self.sample(logits, seqs)
