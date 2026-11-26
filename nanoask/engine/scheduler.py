"""
Continuous-batching scheduler, adapted from nano-vllm/engine/scheduler.py.

Every engine step the scheduler picks a batch:
  * prefill first: admit waiting requests while the token budget and free KV blocks
    allow (the first one may be chunked if its prompt exceeds the budget);
  * otherwise decode: one token for every running sequence.
Requests join and leave the batch at token granularity: a request that finishes frees
its blocks immediately and a newly arrived one is admitted on the next step. Under
memory pressure the most recently admitted running sequence is preempted (blocks freed,
put back at the head of the queue, recomputed later).
"""
from __future__ import annotations

from collections import deque

from nanoask.engine.block_manager import BlockManager
from nanoask.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int, max_model_len: int,
                 num_kvcache_blocks: int, block_size: int, eos: int):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.eos = eos
        self.block_size = block_size
        self.block_manager = BlockManager(num_kvcache_blocks, block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        assert seq.num_tokens < self.max_model_len, "prompt longer than max_model_len"
        needed = -(-min(self.max_model_len, seq.num_tokens + seq.sampling_params.max_tokens) // self.block_size)
        total = self.block_manager.num_free_blocks + len(self.block_manager.used_block_ids)
        if needed > total:
            raise ValueError(f"request needs up to {needed} KV blocks but the cache only has {total}; "
                             "free GPU memory or lower max_model_len / max_tokens")
        self.waiting.append(seq)

    def abort(self, seq: Sequence):
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.running:
            self.running.remove(seq)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = "abort"

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled: list[Sequence] = []
        num_batched = 0

        # ---- prefill
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:  # continuing a chunked prefill
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled:  # only the first seq may be chunked
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled.append(seq)
        if scheduled:
            return scheduled, True

        # ---- decode
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        assert scheduled, "KV cache too small to hold even one sequence"
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool) -> list[tuple[Sequence, int]]:
        """Apply sampled tokens. Returns (seq, new_token) pairs for streaming."""
        emitted = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue  # chunked prefill not done yet: discard the sample
            seq.append_token(token_id)
            sp = seq.sampling_params
            reason = None
            if not sp.ignore_eos and (token_id == self.eos or token_id in sp.stop_token_ids):
                reason = "stop"
            elif seq.num_completion_tokens >= sp.max_tokens:
                reason = "length"
            elif seq.num_tokens >= self.max_model_len:
                reason = "length"
            emitted.append((seq, token_id))
            if reason:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = reason
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return emitted
