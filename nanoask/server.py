"""
NanoAsk server: upload documents, ask questions, get grounded streaming answers.

  POST /ingest          multipart file upload (.pdf .md .txt .html)
  POST /ingest/text     {"title", "text"}
  POST /query           {"question", "history"?, "k"?, "max_tokens"?, "temperature"?, "stream"?, "mode"?}
  POST /generate        {"messages" | "prompt", ...}  plain chat, no retrieval
  GET  /stats           KV-cache / scheduler / index statistics
  GET  /                minimal web UI

All concurrent /query and /generate requests share one continuously-batched engine.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from nanoask.engine.llm_engine import AsyncLLMEngine, EngineConfig, LLMEngine
from nanoask.engine.sequence import SamplingParams
from nanoask.model.gpt import GPT, GPTConfig
from nanoask.model.loader import load_checkpoint, pick_device
from nanoask.model.tokenizer import Tokenizer
from nanoask.rag.pipeline import RAGPipeline, RetrievalConfig

log = logging.getLogger("nanoask")
ROOT = Path(__file__).resolve().parent.parent


class QueryRequest(BaseModel):
    question: str
    history: list[dict] | None = None
    k: int | None = None
    max_tokens: int = 200
    temperature: float = 0.0
    top_k: int | None = 50
    stream: bool = False
    # "auto": LLM answer, checked against the passages; falls back to extractive if ungrounded
    # "llm": raw LLM answer (still reports the grounding check) | "extractive": no LLM
    mode: str = "auto"


class GenerateRequest(BaseModel):
    messages: list[dict] | None = None
    prompt: str | None = None
    max_tokens: int = 200
    temperature: float = 0.7
    top_k: int | None = 50
    stream: bool = False


class TextDoc(BaseModel):
    title: str
    text: str


def build_app(engine: LLMEngine, rag: RAGPipeline, model_info: dict) -> FastAPI:
    app = FastAPI(title="NanoAsk")
    aeng = AsyncLLMEngine(engine)
    tok = engine.tokenizer
    max_len = engine.config.max_model_len

    # if the model was trained by scripts/train_tiny.py, prompt it exactly as it was trained
    train_meta = model_info.get("train") or {}
    ctx_chars_default = train_meta.get("ctx_chars", 6000)

    def finalize(req, hits, raw_answer, extractive) -> dict:
        g = rag.ground_answer(req.question, raw_answer, hits)
        use_fallback = req.mode == "auto" and not g["grounded"]
        return {"answer": extractive if use_fallback else (g["answer"] if g["supported"] else raw_answer.strip()),
                "llm_answer": raw_answer.strip(), "grounded": g["grounded"], "supported": g["supported"],
                "relevant": g["relevant"], "fallback": use_fallback}

    def fit_prompt(question, hits, history, max_tokens) -> list[int]:
        budget = max_len - max_tokens - 1
        chars = ctx_chars_default
        while True:
            ids = tok.render_for_completion(rag.build_messages(question, hits, history, chars))
            if len(ids) <= budget or chars < 200:
                return ids[-budget:]
            chars = int(chars * 0.7)

    async def run_generation(prompt_ids, sp):
        """Yields text deltas (decoding the whole completion each time keeps multi-byte chars intact)."""
        out_ids, sent = [], ""
        async for t, reason in aeng.stream(prompt_ids, sp):
            if t not in tok.stop_token_ids:
                out_ids.append(t)
            text = tok.decode(out_ids)
            if not text.endswith("�") and len(text) > len(sent):
                yield text[len(sent):], None
                sent = text
            if reason:
                yield "", {"finish_reason": reason, "completion_tokens": len(out_ids)}

    def sse(obj) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def sampling(req) -> SamplingParams:
        return SamplingParams(temperature=req.temperature, top_k=req.top_k, max_tokens=req.max_tokens,
                              stop_token_ids=tok.stop_token_ids)

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...)):
        res = []
        for f in files:
            data = await f.read()
            try:
                n = rag.ingest_file(f.filename, data)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(400, f"{f.filename}: {e}")
            res.append({"file": f.filename, "chunks_added": n})
        return {"ingested": res, "index": rag.stats()}

    @app.get("/documents")
    async def documents():
        return {"documents": rag.list_documents()}

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str):
        n = rag.delete_document(doc_id)
        if not n:
            raise HTTPException(404, "no such document")
        return {"deleted_chunks": n, "index": rag.stats()}

    @app.post("/ingest/text")
    async def ingest_text(doc: TextDoc):
        return {"chunks_added": rag.ingest_text(doc.title, doc.text), "index": rag.stats()}

    @app.post("/query")
    async def query(req: QueryRequest):
        t0 = time.perf_counter()
        hits = rag.retrieve(req.question, req.k)
        passages = [h.to_dict(i) for i, h in enumerate(hits, 1)]
        t_ret = time.perf_counter() - t0
        extractive = rag.extractive_answer(req.question, hits)
        if req.mode == "extractive":
            return {"answer": extractive, "passages": passages, "retrieval_ms": round(t_ret * 1000, 1)}
        prompt_ids = fit_prompt(req.question, hits, req.history, req.max_tokens)
        sp = sampling(req)
        if req.stream:
            async def events():
                yield sse({"type": "passages", "passages": passages, "retrieval_ms": round(t_ret * 1000, 1),
                           "prompt_tokens": len(prompt_ids)})
                raw = ""
                async for delta, done in run_generation(prompt_ids, sp):
                    if delta:
                        raw += delta
                        yield sse({"type": "token", "text": delta})
                    if done:
                        yield sse({"type": "done", **done, **finalize(req, hits, raw, extractive),
                                   "extractive_answer": extractive,
                                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
            return StreamingResponse(events(), media_type="text/event-stream")
        answer, info = "", {}
        async for delta, done in run_generation(prompt_ids, sp):
            answer += delta
            info = done or info
        return {**finalize(req, hits, answer, extractive), "extractive_answer": extractive, "passages": passages,
                "prompt_tokens": len(prompt_ids), **info,
                "retrieval_ms": round(t_ret * 1000, 1), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        if req.messages:
            ids = tok.render_for_completion(req.messages)
        elif req.prompt is not None:
            ids = tok.encode(req.prompt, prepend="<|bos|>")
        else:
            raise HTTPException(400, "messages or prompt required")
        ids = ids[-(max_len - req.max_tokens - 1):]
        sp = sampling(req)
        if req.stream:
            async def events():
                async for delta, done in run_generation(ids, sp):
                    if delta:
                        yield sse({"type": "token", "text": delta})
                    if done:
                        yield sse({"type": "done", **done})
            return StreamingResponse(events(), media_type="text/event-stream")
        text, info = "", {}
        async for delta, done in run_generation(ids, sp):
            text += delta
            info = done or info
        return {"text": text, **info}

    @app.get("/stats")
    async def stats():
        st = engine.stats
        return {
            "model": model_info,
            "kv_cache": engine.kv_stats(),
            "engine": {
                "steps": st.steps, "max_batch_size": st.max_batch_size,
                "prefill_tok_per_s": round(st.prefill_tokens / st.prefill_time, 1) if st.prefill_time else None,
                "decode_tok_per_s": round(st.decode_tokens / st.decode_time, 1) if st.decode_time else None,
            },
            "index": rag.stats(),
        }

    @app.get("/", response_class=HTMLResponse)
    async def ui():
        return (ROOT / "nanoask" / "static" / "index.html").read_text()

    app.state.async_engine = aeng
    return app


# ---------------------------------------------------------------------------- bootstrap
def random_model(docs_dir: str | None, device: torch.device) -> tuple[GPT, Tokenizer]:
    """No checkpoint: tiny randomly initialised nanochat model + tokenizer trained on the docs.
    Exercises the full serving stack, but the generated text is noise."""
    texts = [p.read_text(errors="replace") for p in Path(docs_dir or ROOT / "data/sample_docs").glob("*")
             if p.is_file()] or ["hello world"] * 10
    tok = Tokenizer.train_from_iterator(iter(texts * 4), vocab_size=1024)
    cfg = GPTConfig(sequence_len=1024, vocab_size=tok.get_vocab_size(), n_layer=4, n_head=4, n_kv_head=2,
                    n_embd=128, window_pattern="SSSL")
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights()
    return model.to(device).eval(), tok


def main(argv=None):
    p = argparse.ArgumentParser(description="NanoAsk: RAG + paged-KV continuous-batching server")
    p.add_argument("--checkpoint", help="nanochat checkpoint dir (model_*.pt + meta_*.json)")
    p.add_argument("--tokenizer-dir", help="dir with tokenizer.pkl (default: <checkpoint>/tokenizer or ~/.cache/nanochat/tokenizer)")
    p.add_argument("--step", type=int)
    p.add_argument("--device")
    p.add_argument("--docs", help="directory of documents to ingest at startup")
    p.add_argument("--store", default=str(ROOT / "data/index.json"), help="where ingested chunks are persisted")
    p.add_argument("--embedder", default="lsa", help="lsa | st:<sentence-transformers model>")
    p.add_argument("--reranker", default="feature", help="feature | ce:<cross-encoder model> | none")
    p.add_argument("--fusion", default="rrf", choices=["rrf", "alpha"])
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--kv-cache-mb", type=float, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = pick_device(args.device)
    default_ckpt = ROOT / "checkpoints" / "tiny"
    ckpt = args.checkpoint or (str(default_ckpt) if default_ckpt.exists() else None)
    if ckpt:
        model, tok, meta = load_checkpoint(ckpt, args.tokenizer_dir, args.step, str(device))
        info = {"source": ckpt, "config": model.config.__dict__}
        if "train" in meta:  # trained by scripts/train_tiny.py: reuse its prompt settings
            info["train"] = {k: meta["train"][k] for k in ("k", "ctx_chars") if k in meta["train"]}
    else:
        log.warning("No checkpoint given: using a RANDOM tiny model. Answers will be noise; "
                    "use mode=extractive, run scripts/train_tiny.py, or pass --checkpoint.")
        model, tok = random_model(args.docs, device)
        info = {"source": "random-init", "config": model.config.__dict__}
    info["device"] = str(device)
    info["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    engine = LLMEngine(model, tok, EngineConfig(
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=min(args.max_model_len, model.config.sequence_len), block_size=args.block_size,
        kv_cache_mb=args.kv_cache_mb, gpu_memory_utilization=args.gpu_memory_utilization))
    log.info("KV cache: %s", engine.kv_stats())

    rag = RAGPipeline(args.embedder, args.reranker, retrieval=RetrievalConfig(fusion=args.fusion,
                                                                               final_k=info.get("train", {}).get("k", 4)),
                      store_path=args.store)
    if args.docs:
        for f in sorted(Path(args.docs).glob("*")):
            if f.is_file():
                n = rag.ingest_file(f.name, f.read_bytes())
                log.info("ingested %s: %d new chunks", f.name, n)
    log.info("index: %s", rag.stats())

    import uvicorn
    uvicorn.run(build_app(engine, rag, info), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
