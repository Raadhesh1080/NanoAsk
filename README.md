# NanoAsk

A small, readable document Q&A server that combines three reference projects:

| Layer | Borrowed from | What it does here |
|---|---|---|
| **Model**: transformer, attention, tokenizer, sampling | [karpathy/nanochat](https://github.com/karpathy/nanochat) | nanochat's GPT (RoPE, QK-norm, GQA, sliding-window attention, ReLU², value embeddings, smear gate, logit soft-cap) and its rustbpe/tiktoken BPE tokenizer with the `<\|user_start\|>…<\|assistant_end\|>` chat format. Parameter names are identical, so **real nanochat checkpoints load with `strict=True`**. |
| **Serving**: paged KV cache, continuous batching | [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | `Sequence` / `BlockManager` (block tables, hash-chained prefix caching, ref counts) / `Scheduler` (prefill-first, chunked prefill, decode, preemption) / `ModelRunner` (flat varlen batch, slot mapping). An async wrapper lets many HTTP clients share one batch. |
| **RAG**: ingest, hybrid search, rerank | [deadjoe/askme](https://github.com/deadjoe/askme) | PDF/HTML/Markdown/text loading, askme's recursive chunker with overlap, BM25 + HNSW dense search, RRF / alpha fusion, a second-stage reranker, and a citation-style prompt. |

```
              ┌──────────── RAG (askme) ────────────┐
 question ──► │ BM25 ─┐                              │
              │       ├─► RRF ─► rerank ─► top-k ────┼─► prompt [1]..[k] ─┐
              │ HNSW ─┘                              │                    │
              └──────────────────────────────────────┘                    ▼
                         ┌──────── engine (nano-vllm) ─────────┐  ┌── model (nanochat) ──┐
 other users' requests ─►│ scheduler ─► block manager ─► runner │─►│ GPT.forward_paged    │─► streamed tokens
                         │  (continuous batching)  (paged KV)   │  │ + BPE tokenizer      │
                         └──────────────────────────────────────┘  └──────────────────────┘
```

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest tests -q                       # 20 tests, ~45 s on CPU
python scripts/train_tiny.py --steps 3000 --n-layer 6   # optional: ~20 min CPU demo model (see below)
python scripts/eval_qa.py                       # QA accuracy on 16 hand-written questions
python -m nanoask.server --docs data/sample_docs
# open http://127.0.0.1:8000, or:
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question":"How much is the hotel reimbursement per night?"}'
```

### Using a real nanochat model (recommended for real answers)

Train or download a nanochat SFT checkpoint (e.g. with nanochat's `speedrun.sh` on a GPU node), then:

```bash
python -m nanoask.server \
  --checkpoint ~/.cache/nanochat/chatsft_checkpoints/d20 \
  --tokenizer-dir ~/.cache/nanochat/tokenizer \
  --device cuda --docs path/to/your/docs
```

On CUDA the KV cache is sized nano-vllm style: whatever memory is left under `--gpu-memory-utilization` after loading the weights. On CPU it uses `--kv-cache-mb`.

### Neural embedder / reranker (askme's BGE models)

```bash
pip install -e ".[neural]"
python -m nanoask.server --embedder st:BAAI/bge-m3 --reranker ce:BAAI/bge-reranker-v2-m3 ...
```

## API

| Endpoint | Body | Notes |
|---|---|---|
| `POST /ingest` | multipart `files` | `.pdf .md .txt .html`; re-uploading the same document is a no-op |
| `POST /ingest/text` | `{"title","text"}` | |
| `POST /query` | `{"question", "history"?, "k"?, "max_tokens"?, "temperature"?, "stream"?, "mode"?}` | returns `answer`, `llm_answer`, `grounded`, `fallback`, `passages` (with BM25 and dense ranks), `extractive_answer`. `stream:true` gives SSE events `passages` → `token`… → `done` (which carries the final checked answer). Modes: `auto` (default; LLM answer checked against the passages, falls back to extractive if ungrounded), `llm` (raw model output), `extractive` (no LLM). `history` enables follow-up questions. |
| `GET /documents` | | indexed documents with chunk counts |
| `DELETE /documents/{doc_id}` | | removes a document and rebuilds the indexes |
| `POST /generate` | `{"messages"}` or `{"prompt"}` | plain chat, no retrieval |
| `GET /stats` | | KV blocks used/free, prefix-cache hits, preemptions, batch sizes, tok/s, index info |

## How the pieces map to the source repos

**nano-vllm → `nanoask/engine/`**
- `block_manager.py` is nano-vllm's allocator almost line for line: fixed-size blocks, `block_table` per sequence, xxhash replaced by blake2b for hash-chained **prefix caching**. RAG prompts share a long system prefix, so those blocks are computed once and reused across users (`prefix_cache_block_hits` in `/stats`).
- `scheduler.py` is the same prefill-first continuous-batching policy, with chunked prefill for the first waiting sequence and preemption on block exhaustion. Additions: stop tokens, `abort` for disconnected clients, max-length guard.
- `model_runner.py` builds nano-vllm's flat varlen batch (`slot_mapping`, `cu_seqlens_q`, `block_tables`). Prefill and decode share one path.
- `paged_attention.py` replaces the Triton store kernel and `flash_attn_varlen_func` / `flash_attn_with_kvcache` with plain PyTorch (scatter into slots, gather by block table, one padded masked SDPA call). This runs on CPU, GPU and MPS. On GPU it is slower than flash-attn; swapping flash-attn back in is the obvious production step.
- `llm_engine.py` provides `LLMEngine.generate()` for offline batches and `AsyncLLMEngine`, a background thread stepping the engine while HTTP handlers stream from per-request queues.

**nanochat → `nanoask/model/`**
- `gpt.py` is nanochat's architecture and init, plus `forward_paged()`. nanochat's smear gate needs the previous token's embedding; it is recomputed from `prev_ids`, so the paged cache needs no extra per-sequence state.
- `tokenizer.py`: rustbpe training, tiktoken inference, nanochat special tokens and conversation rendering.
- `loader.py` reads nanochat's `model_XXXXXX.pt` + `meta_XXXXXX.json` + `tokenizer.pkl`.

**askme → `nanoask/rag/`**
- `documents.py`: loaders and askme's recursive separator chunker with overlap.
- `bm25.py`: Okapi BM25. `hnsw.py`: a from-scratch HNSW (multi-layer graph, diversity-heuristic neighbour selection, ef beam search). askme delegates HNSW to Milvus/Weaviate/Qdrant.
- `fusion.py`: askme's `reciprocal_rank_fusion` and `alpha_fusion`.
- `rerank.py`: a cross-encoder (askme's BGE reranker) when available; offline default is a pairwise feature reranker (idf coverage, term proximity, phrase match, dense cosine).
- `embeddings.py`: BGE via sentence-transformers when available; offline default is LSA (TF-IDF + truncated SVD).
- `pipeline.py`: ingest → hybrid retrieve → rerank → numbered-citation prompt, plus a model-free extractive answer and `ground_answer()`, a faithfulness check. Each answer sentence must be supported by a passage (≥70% of its terms); its citation is rewritten to the passage that actually supports it; and it must be at least half as on-topic as the best sentence available. New documents are folded into the existing embedding and appended to HNSW; a full refit happens only after the corpus grows 25%.

## Verified behaviour (tests/)

- **Real nanochat compatibility** (`test_nanochat_compat.py`): a model built and saved with nanochat's own `GPT` and `checkpoint_manager`, including the `_orig_mod.` prefix from `torch.compile`, plus a tokenizer trained with nanochat's own `RustBPETokenizer`. Both load through NanoAsk with logits equal to 1e-5. Batched paged generation is token-identical to nanochat's `GPT.generate`, and chat rendering is byte-identical. The test needs a nanochat checkout (`NANOCHAT_REPO`) and is skipped otherwise.
- **Paged == dense**: greedy outputs from the paged, batched, chunked-prefill engine are token-identical to nanochat's plain dense forward, including sliding windows and GQA. Single-step logits match to 1e-4.
- **Prefix caching** reuses blocks across requests and still gives identical outputs.
- **Preemption**: six sequences in a KV pool too small for them all still produce correct outputs, and every block is returned.
- **HNSW** recall@10 > 0.9 against brute force on 2,000 random vectors.
- **Retrieval**: the right document ranks first for five question types on the sample docs.
- **Server**: concurrent `/query` calls land in the same batch (`max_batch_size > 1`). Streaming, extractive and auto fallback, follow-up history, and document list/delete all work.
- **Index maintenance**: small ingests don't refit the embedding; deletes leave BM25, HNSW and the store consistent.
- **Grounding check**: accepts a correct answer and fixes its citation; rejects a copied but off-topic sentence and a hallucination.

`scripts/bench.py` on a 4-core CPU with an 8M-param model and 24 requests: **5.2× throughput** from continuous batching over one-at-a-time serving, and **5× less KV memory** with paging than reserving `max_model_len` per request.

## Honest limitations

- **Answer quality depends on the model.** Without any checkpoint the server uses a random tiny model, so its raw answers are noise; Auto mode then falls back to the extractive answer. `scripts/train_tiny.py --steps 3000 --n-layer 6` (about 20 min on a 4-core CPU) trains a 3.4M-param nanochat-architecture model to copy the relevant sentence from the context and cite it. Results from `scripts/eval_qa.py` on 16 hand-written questions over the sample docs:

  | retrieval@1 | raw model | Auto mode (checked + fallback) | extractive only | model answer grounded |
  |---|---|---|---|---|
  | 100% | 69% | 88% | 94% | 75% |

  The grounding check had one false accept: a related but different sentence for "Why does the KV cache use so much memory?". The model is trained only on the given documents, so for new documents retrain it, or better, use a real nanochat d20/d32 SFT checkpoint.
- No tensor parallelism and no CUDA graphs (both are in nano-vllm and dropped here for clarity and CPU support).
- With the default LSA embedder, chunks folded in between refits only use vocabulary seen at the last fit; the next refit (after 25% growth) picks up new terms. BM25 always indexes every term immediately.
- Deleting a document rebuilds the indexes (HNSW has no cheap delete).
- GPU paths (`--device cuda`, bf16, memory-based KV sizing) are written but untested here (no GPU in this environment). flash-attn and CUDA graphs are not used.
- The pure-Python HNSW is for clarity, not speed. For millions of vectors use hnswlib/FAISS/Milvus, as askme does.
