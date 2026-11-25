"""
LLMEngine (offline, synchronous) and AsyncLLMEngine (online, serves many users at once).
Adapted from nano-vllm/engine/llm_engine.py.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from nanoask.engine.model_runner import ModelRunner
from nanoask.engine.scheduler import Scheduler
from nanoask.engine.sequence import SamplingParams, Sequence
from nanoask.model.gpt import GPT
from nanoask.model.tokenizer import Tokenizer


@dataclass
class EngineConfig:
    max_num_seqs: int = 32               # max concurrent sequences in a batch
    max_num_batched_tokens: int = 2048   # per-step token budget (prefill chunking)
    max_model_len: int = 2048
    block_size: int = 16                 # tokens per KV block
    num_kvcache_blocks: int | None = None
    kv_cache_mb: float = 256             # CPU budget; on CUDA gpu_memory_utilization is used
    gpu_memory_utilization: float = 0.9
    seed: int = 0


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    max_batch_size: int = 0
    batch_sizes: list[int] = field(default_factory=list)


class LLMEngine:
    def __init__(self, model: GPT, tokenizer: Tokenizer, config: EngineConfig | None = None):
        self.config = config = config or EngineConfig()
        self.tokenizer = tokenizer
        config.max_model_len = min(config.max_model_len, model.config.sequence_len * 10)
        Sequence.block_size = config.block_size
        self.runner = ModelRunner(model, config.block_size, config.num_kvcache_blocks,
                                  config.kv_cache_mb, config.gpu_memory_utilization, config.seed)
        self.scheduler = Scheduler(config.max_num_seqs, config.max_num_batched_tokens, config.max_model_len,
                                   self.runner.num_kvcache_blocks, config.block_size,
                                   eos=tokenizer.encode_special("<|assistant_end|>"))
        self.stats = EngineStats()

    def add_request(self, prompt: str | list[int], sp: SamplingParams | None = None) -> Sequence:
        ids = self.tokenizer.encode(prompt, prepend="<|bos|>") if isinstance(prompt, str) else prompt
        seq = Sequence(ids, sp)
        self.scheduler.add(seq)
        return seq

    def step(self) -> list[tuple[Sequence, int]]:
        seqs, is_prefill = self.scheduler.schedule()
        t0 = time.perf_counter()
        token_ids = self.runner.run(seqs, is_prefill)
        dt = time.perf_counter() - t0
        st = self.stats
        st.steps += 1
        st.max_batch_size = max(st.max_batch_size, len(seqs))
        st.batch_sizes.append(len(seqs))
        if is_prefill:
            st.prefill_tokens += sum(s.num_scheduled_tokens for s in seqs)
            st.prefill_time += dt
        else:
            st.decode_tokens += len(seqs)
            st.decode_time += dt
        return self.scheduler.postprocess(seqs, token_ids, is_prefill)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts: list[str | list[int]], sp: SamplingParams | list[SamplingParams] | None = None) -> list[dict]:
        sps = sp if isinstance(sp, list) else [sp] * len(prompts)
        seqs = [self.add_request(p, s) for p, s in zip(prompts, sps)]
        while not self.is_finished():
            self.step()
        return [self._output(s) for s in seqs]

    def _output(self, seq: Sequence) -> dict:
        ids = [t for t in seq.completion_token_ids if t not in self.tokenizer.stop_token_ids]
        return {"text": self.tokenizer.decode(ids), "token_ids": seq.completion_token_ids,
                "finish_reason": seq.finish_reason}

    def kv_stats(self) -> dict:
        bm = self.scheduler.block_manager
        n = self.runner.num_kvcache_blocks
        return {
            "block_size": self.config.block_size,
            "num_blocks": n,
            "used_blocks": n - bm.num_free_blocks,
            "utilization": round((n - bm.num_free_blocks) / n, 4),
            "kv_cache_mb": round(self.runner.kv_cache_bytes / 2**20, 2),
            "prefix_cache_block_hits": bm.prefix_hits,
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "preemptions": self.scheduler.num_preemptions,
        }


class AsyncLLMEngine:
    """
    Runs LLMEngine.step() in a background thread. Web handlers submit requests from
    any coroutine and receive a token stream; all concurrent requests are merged into
    the same continuous batch by the scheduler.
    """

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="llm-engine")
        self._thread.start()

    def _loop(self):
        while not self._stop:
            with self._lock:
                idle = self.engine.is_finished()
            if idle:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            with self._lock:
                emitted = self.engine.step()
            for seq, tok in emitted:
                loop, q = self._streams.get(seq.seq_id, (None, None))
                if q is None:
                    continue
                loop.call_soon_threadsafe(q.put_nowait, (tok, seq.finish_reason))
                if seq.is_finished:
                    self._streams.pop(seq.seq_id, None)

    async def stream(self, prompt_ids: list[int], sp: SamplingParams):
        """Yields (token_id, finish_reason|None)."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        seq = Sequence(prompt_ids, sp)
        with self._lock:
            self._streams[seq.seq_id] = (loop, q)
            self.engine.scheduler.add(seq)
        self._wake.set()
        try:
            while True:
                tok, reason = await q.get()
                yield tok, reason
                if reason:
                    return
        finally:
            if not seq.is_finished:  # client disconnected
                with self._lock:
                    self.engine.scheduler.abort(seq)
                self._streams.pop(seq.seq_id, None)

    def shutdown(self):
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2)
