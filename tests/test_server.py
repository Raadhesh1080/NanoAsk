import asyncio
import json
from pathlib import Path

import httpx
import pytest
import torch

from nanoask.engine.llm_engine import EngineConfig, LLMEngine
from nanoask.rag.pipeline import RAGPipeline
from nanoask.server import build_app, random_model

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(scope="module")
def app():
    model, tok = random_model(str(DOCS), torch.device("cpu"))
    engine = LLMEngine(model, tok, EngineConfig(max_model_len=1024, block_size=16, kv_cache_mb=32))
    rag = RAGPipeline()
    app = build_app(engine, rag, {"source": "test"})
    yield app
    app.state.async_engine.shutdown()


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120)


def test_ingest_query_concurrency(app):
    async def run():
        async with client(app) as c:
            files = [("files", (p.name, p.read_bytes())) for p in sorted(DOCS.glob("*.md"))]
            r = await c.post("/ingest", files=files)
            assert r.status_code == 200 and r.json()["index"]["documents"] == 4

            qs = ["How much is the hotel reimbursement?", "What is HNSW?", "What is prefix caching?",
                  "How many sick days?", "What is rotary embedding?", "What is BM25?"]
            body = lambda q: {"question": q, "max_tokens": 24, "temperature": 0}
            rs = await asyncio.gather(*[c.post("/query", json=body(q)) for q in qs])
            for q, r in zip(qs, rs):
                j = r.json()
                assert r.status_code == 200, j
                assert j["passages"] and j["completion_tokens"] > 0
            assert "180 euros" in rs[0].json()["extractive_answer"]

            st = (await c.get("/stats")).json()
            assert st["engine"]["max_batch_size"] > 1           # requests were batched together
            assert st["kv_cache"]["used_blocks"] == 0           # all blocks returned
            assert st["kv_cache"]["prefix_cache_block_hits"] > 0  # shared system-prompt blocks reused

            # streaming
            async with c.stream("POST", "/query", json={**body("What is HNSW?"), "stream": True}) as r:
                events = [json.loads(l[6:]) async for l in r.aiter_lines() if l.startswith("data: ")]
            assert events[0]["type"] == "passages" and events[-1]["type"] == "done"

            r = await c.post("/query", json={"question": "sick days", "mode": "extractive"})
            assert "10 paid sick days" in r.json()["answer"]

            # auto mode: the random model's output is never grounded -> extractive fallback
            j = (await c.post("/query", json={**body("How many sick days do employees get?"), "mode": "auto"})).json()
            assert j["fallback"] and not j["grounded"] and "10 paid sick days" in j["answer"]
            j = (await c.post("/query", json={**body("How many sick days?"), "mode": "llm"})).json()
            assert not j["fallback"] and j["answer"] == j["llm_answer"] or j["supported"]

            # follow-up with history
            hist = [{"role": "user", "content": "What is HNSW?"}, {"role": "assistant", "content": "A graph index."}]
            r = await c.post("/query", json={**body("How is it searched?"), "history": hist})
            assert r.status_code == 200

            # document management
            docs = (await c.get("/documents")).json()["documents"]
            assert len(docs) == 4
            pol = next(d for d in docs if "Northwind" in d["title"])
            r = await c.delete(f"/documents/{pol['doc_id']}")
            assert r.json()["index"]["documents"] == 3
            assert (await c.delete("/documents/missing")).status_code == 404
    asyncio.run(run())
