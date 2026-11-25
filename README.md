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
python -m pytest tests -q                       # 23 tests, ~1 min on CPU
python -m nanoask.server --docs data/sample_docs
# open http://127.0.0.1:8000, or:
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question":"How much is the hotel reimbursement per night?"}'
```

Evaluation and benchmarks (see [Results](#results)):

```bash
python scripts/build_corpus.py                  # real training/distractor text: READMEs of 225 PyPI packages
python scripts/eval_qa.py                       # 88 handwritten questions over 8 documents, with leak check
python scripts/eval_retrieval.py                # HNSW recall on real chunk embeddings
python scripts/bench.py                         # batching throughput + KV memory
python scripts/train_tiny.py --steps 5000 --n-layer 6 --k 2 --ctx-chars 1200   # demo model, ~50 min CPU
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
- `pipeline.py`: ingest → hybrid retrieve → rerank → numbered-citation prompt, plus a model-free extractive answer and `ground_answer()`, a faithfulness check. Each answer sentence must be supported by a passage (≥70% of its terms); its citation is rewritten to the passage that actually supports it; and it must be at least as on-topic as the best sentence the extractive answer would use. New documents are folded into the existing embedding and appended to HNSW; a full refit happens only after the corpus grows 25%.

## Verified behaviour (tests/)

- **Real nanochat compatibility** (`test_nanochat_compat.py`): a model built and saved with nanochat's own `GPT` and `checkpoint_manager`, including the `_orig_mod.` prefix from `torch.compile`, plus a tokenizer trained with nanochat's own `RustBPETokenizer`. Both load through NanoAsk with logits equal to 1e-5. Batched paged generation is token-identical to nanochat's `GPT.generate`, and chat rendering is byte-identical. The test needs a nanochat checkout (`NANOCHAT_REPO`) and is skipped otherwise.
- **Paged == dense**: greedy outputs from the paged, batched, chunked-prefill engine are token-identical to nanochat's plain dense forward, including sliding windows and GQA. Single-step logits match to 1e-4. `NANOASK_TEST_DEVICE=cuda` runs these on a GPU.
- **bf16** (the CUDA dtype): paged logits stay within tolerance of the dense forward; batched generation frees all KV blocks.
- **Prefix caching** reuses blocks across requests and still gives identical outputs.
- **Preemption**: six sequences in a KV pool too small for them all still produce correct outputs, and every block is returned.
- **HNSW** recall@10 > 0.9 on 2,000 random vectors and > 0.95 on real LSA embeddings of installed-package READMEs.
- **Server**: concurrent `/query` calls land in the same batch. Streaming, extractive and auto fallback, follow-up history, and document list/delete all work.
- **Index maintenance**: small ingests don't refit the embedding; deletes leave BM25, HNSW and the store consistent.
- **Grounding check**: accepts a correct answer and fixes its citation; rejects a copied but off-topic sentence and a hallucination.

## Results

All numbers below were measured on a 4-core x86 CPU (no GPU), torch 2.15, with nothing else running.

### Question answering (`scripts/eval_qa.py`)

88 handwritten questions over the 8 documents in `data/sample_docs` (company handbook, library rules, clinic information, solar battery manual, a nature article, and three ML explainers). The expected answer text is in `data/eval/qa.jsonl`.

**The demo model never sees these documents.** It and its tokenizer are trained only on READMEs of 225 PyPI packages (`scripts/build_corpus.py`). The eval script checks every test sentence against fingerprints of the training sentences: **0 of 145 overlap**.

| | ret@1 | ret@k | model alone | **Auto** | extractive, 1 sentence | extractive, ≤3 sentences |
|---|---|---|---|---|---|---|
| 8 test docs in the index | 95% | 100% | **0%** | **93%** | 85% | 93% |
| + 225 unrelated READMEs as distractors (1,851 chunks) | 93% | 97% | 0% | 90% | 84% | 90% |

- **Retrieval works; the tiny generator does not.** The demo model (3.5M params, 5,000 steps, ~50 min on CPU) learned the citation format and README-style language, but not reading comprehension: on unseen documents it produces fluent-looking nonsense. Training on longer contexts (3 passages) stalled at the same point. A model this small cannot learn to read general text in a CPU-hour; for real generated answers, use a nanochat SFT checkpoint (`--checkpoint`).
- **Earlier versions of this README reported 69% for the model. That was a leak:** the model was trained on the same documents it was tested on. Re-scored on this 88-question set, that leaky model answered 55% of the questions on the 4 documents it had been trained on (25–82% per document) and **0% on the 4 documents it had never seen**.
- **Auto mode vs extractive.** Auto keeps the model's answer only if it passes the grounding check, and otherwise serves the extractive answer. The check rejected all 88 of the clean model's answers, so Auto equals extractive (93%) and the model neither helps nor hurts. Auto used to score *below* extractive (88% vs 94% on the old 16 questions) for two reasons:
  1. The relevance rule was lenient. It kept a model answer that was only half as on-topic as the best available sentence, which let a related-but-wrong sentence through. The rule now requires the model's answer to be **at least as relevant as the sentence extraction would pick**. On the leaky model this raised Auto from 89% to 92%; the remaining gap is a single tie the model broke wrongly.
  2. The comparison wasn't like for like. The extractive answer returns up to 3 sentences, which gives it more chances to contain the fact than a one-sentence model answer. Restricted to one sentence, extraction scores 85%.

  So with this demo model, the language model adds nothing over extraction; Auto exists so that a weak or hallucinating model can't make answers worse. The model only becomes useful with a checkpoint that can actually read, at which point the same eval script measures it (`--checkpoint ...`).

### Retrieval (`scripts/eval_retrieval.py`)

HNSW recall against exact cosine search, on real LSA embeddings (256-d) of 1,851 chunks (the PyPI READMEs plus the test documents). Queries: the 88 questions plus 300 sentences sampled from the corpus. k = 10, M = 16, ef_construction = 200.

| index | recall@10 | ms / query |
|---|---|---|
| NanoAsk HNSW, ef_search = 16 | 0.972 | 0.46 |
| NanoAsk HNSW, ef_search = 32 | 0.993 | 0.51 |
| **NanoAsk HNSW, ef_search = 64 (default)** | **0.999** | 0.84 |
| NanoAsk HNSW, ef_search = 128 | 1.000 | 1.47 |
| hnswlib (C++), ef = 16 / 64 | 0.935 / 0.989 | 0.03 / 0.06 |
| brute force (numpy) | 1.000 | 0.05 |

At this size, brute force is faster than either HNSW. Approximate search only pays off at much larger collections, and there the pure-Python implementation should be swapped for hnswlib/FAISS.

### Serving (`scripts/bench.py`)

CPU preset: random-weight nanochat GPT (6 layers, 256-d, 8.0M params; throughput doesn't depend on the weight values), 24 requests, prompts 50–300 tokens, outputs 32–64 tokens, `max_model_len = 1024`, KV block size 16.

| | output tok/s |
|---|---|
| one request at a time | 219–258 |
| continuous batching | 868–910 |

Continuous batching gave **3.5–4.0×** the throughput across two runs (the range is run-to-run noise on a shared CPU; the latest run is in `results/bench_cpu.md`).

**KV memory with all 24 requests resident: 28.7 MB paged vs 144.0 MB if every request reserves `max_model_len = 1024` tokens, i.e. 5.0× less.** The requests use 197 tokens each on average, so the saving scales with `max_model_len` ÷ actual length: with `max_model_len = 2048` the same workload would need 10× less.

### GPU results

Not yet measured. `notebooks/gpu_benchmark.ipynb` runs the GPU correctness tests, `bench.py --preset gpu` (12-layer/768-d model, 128 requests, prompts 100–1000 tokens, `max_model_len = 2048`; the one-at-a-time baseline is timed on 16 of the 128 requests, since it reports a rate) and the QA eval on a free Colab or Kaggle T4, and prints a table to paste here.

## Honest limitations

- **The bundled demo model can't answer questions** (see Results). Auto mode falls back to extractive answers, so the server is still useful, but generated answers need a real nanochat checkpoint.
- **The GPU path is untested** until the notebook above has been run: `--device cuda`, bf16 weights and KV cache, and memory-based KV sizing. The bf16 code path itself is tested on CPU. flash-attn and CUDA graphs are not used.
- No tensor parallelism (nano-vllm has it; dropped for clarity and CPU support).
- With the default LSA embedder, chunks folded in between refits only use vocabulary seen at the last fit; the next refit (after 25% growth) picks up new terms. BM25 always indexes every term immediately.
- Deleting a document rebuilds the indexes (HNSW has no cheap delete).
- The pure-Python HNSW is for clarity, not speed (see the retrieval table).
