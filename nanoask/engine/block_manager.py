"""
Paged KV-cache block allocator with automatic prefix caching.
Adapted from nano-vllm/engine/block_manager.py.

The KV cache is one big pool of fixed-size blocks. Each sequence owns a `block_table`
(list of block ids), so memory is allocated on demand in block_size chunks instead of
reserving max_model_len per request. Full blocks are content-hashed (chained with the
previous block's hash); a new request whose prompt starts with the same tokens - e.g. the
same RAG system prompt or the same retrieved passages - reuses those blocks via ref counts
and skips recomputing them.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from nanoask.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, h: int, token_ids: list[int]):
        self.hash = h
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.prefix_hits = 0  # stats: number of blocks reused from the prefix cache

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        h = hashlib.blake2b(digest_size=8)
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return int.from_bytes(h.digest(), "little")

    @property
    def num_free_blocks(self):
        return len(self.free_block_ids)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)  # keeps its hash: may be revived by a cache hit

    def can_allocate(self, seq: Sequence) -> int:
        """Returns the number of leading prefix-cached blocks, or -1 if memory is short."""
        h, num_cached, num_new = -1, 0, seq.num_blocks
        # never treat the final block as cached: at least one token must be computed
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached += 1
            if block_id in self.used_block_ids:
                num_new -= 1
        if len(self.free_block_ids) < num_new:
            return -1
        return num_cached

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            h = self.compute_hash(seq.block(i), h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        self.prefix_hits += num_cached_blocks
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """Register blocks that just became full so later requests can reuse them."""
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
