# Paged KV Cache and Continuous Batching

## Why the KV cache dominates memory
During autoregressive generation a transformer stores the key and value vectors of every previous token so it does not recompute them. This key/value memory is called the KV cache. Its size per token is 2 x layers x kv_heads x head_dim x bytes per element. For long contexts and many concurrent users the KV cache, not the model weights, is what exhausts GPU memory.

## Paging the cache
Naive servers reserve a contiguous region of max_model_len tokens for every request. Most requests are much shorter, so most of that reservation is wasted, and fragmentation limits how many users can be served. A paged KV cache splits memory into fixed-size blocks, for example 16 or 256 tokens each, exactly like pages in an operating system. Each sequence owns a block table that maps logical token positions to physical blocks. Blocks are allocated only when a sequence actually grows into them, so waste is bounded by one partially filled block per sequence.

## Prefix caching
Every full block is identified by a hash of its tokens chained with the hash of the previous block. When a new request starts with the same tokens as an earlier one, for example a shared system prompt or the same retrieved passages, the block manager finds the existing blocks by hash, increments their reference counts and skips recomputing them. Blocks are returned to the free list when their reference count drops to zero, but keep their hash so they can still be revived.

## Continuous batching
Static batching waits for a whole batch to finish before admitting new work, so short requests wait for long ones. Continuous batching schedules at the granularity of a single decoding step. Each step the scheduler first admits waiting requests for prefill, as long as the token budget and free blocks allow, and otherwise runs one decode step for every running sequence. A finished sequence frees its blocks immediately and a newly arrived request joins on the next step. When memory runs out the scheduler preempts the most recently admitted sequence, frees its blocks and recomputes it later.

## Chunked prefill
A very long prompt can exceed the per-step token budget. Chunked prefill processes the prompt in several pieces across consecutive steps, keeping step latency bounded for everyone else in the batch.
