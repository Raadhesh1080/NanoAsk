from pathlib import Path

import numpy as np

from nanoask.rag.bm25 import BM25Index
from nanoask.rag.documents import ChunkingConfig, RecursiveChunker, load_bytes
from nanoask.rag.fusion import alpha_fusion, reciprocal_rank_fusion
from nanoask.rag.hnsw import HNSWIndex
from nanoask.rag.pipeline import RAGPipeline

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_hnsw_recall_vs_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 32)).astype(np.float32)
    idx = HNSWIndex(32, M=12, ef_construction=100, ef_search=64)
    idx.add(X)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    recall = []
    for q in rng.standard_normal((50, 32)).astype(np.float32):
        truth = set(np.argsort(-(Xn @ (q / np.linalg.norm(q))))[:10])
        got = {i for i, _ in idx.search(q, 10)}
        recall.append(len(truth & got) / 10)
    assert np.mean(recall) > 0.9, np.mean(recall)


def test_bm25_prefers_rare_exact_terms():
    bm = BM25Index()
    bm.add(["the cat sat on the mat", "a dog in the park", "quantum chromodynamics lecture notes", "cats and dogs"])
    top = bm.search("chromodynamics", 2)
    assert top[0][0] == 2 and len(top) == 1
    assert bm.search("cat")[0][0] in (0, 3)


def test_rrf_and_alpha():
    a = [(1, 9.0), (2, 5.0), (3, 1.0)]
    b = [(3, 0.9), (1, 0.8), (4, 0.1)]
    rrf = reciprocal_rank_fusion([a, b], k=60)
    assert rrf[0][0] == 1                       # ranked 1st and 2nd
    assert {d for d, _ in rrf} == {1, 2, 3, 4}
    al = alpha_fusion(b, a, alpha=0.5)
    assert al[0][0] == 1


def test_chunker_respects_size_and_overlaps():
    text = ("Sentence number %d is here. " * 200) % tuple(range(200))
    ch = RecursiveChunker(ChunkingConfig(chunk_size=300, chunk_overlap=60, min_chunk_size=20))
    chunks = ch.split(text)
    assert len(chunks) > 5
    assert all(len(c) <= 300 + 60 + 1 for c in chunks)
    # overlap: each chunk starts with words from the end of the previous one
    assert chunks[1].split()[0] in chunks[0]


def test_html_loader_strips_tags():
    title, text = load_bytes("x.html", b"<html><title>T</title><script>evil()</script><p>Hello <b>there</b></p></html>")
    assert title == "T" and "evil" not in text and "Hello" in text and "<" not in text


def _pipeline(tmp_path=None):
    rag = RAGPipeline(store_path=str(tmp_path / "idx.json") if tmp_path else None)
    for f in sorted(DOCS.glob("*.md")):
        rag.ingest_file(f.name, f.read_bytes())
    return rag


def test_end_to_end_retrieval(tmp_path):
    rag = _pipeline(tmp_path)
    cases = {
        "How much is the hotel reimbursement per night?": "Northwind",
        "How many sick days do employees get?": "Northwind",
        "What does reciprocal rank fusion do?": "Retrieval-Augmented",
        "how does the block manager reuse blocks for a shared system prompt": "Paged KV",
        "What is group-query attention?": "Transformer",
    }
    for q, title_part in cases.items():
        hits = rag.retrieve(q, 3)
        assert title_part in hits[0].chunk.title, (q, [h.chunk.title for h in hits])
    ans = rag.extractive_answer("How much is the hotel reimbursement per night?",
                                rag.retrieve("How much is the hotel reimbursement per night?"))
    assert "180 euros" in ans and "[1]" in ans


def test_persistence_and_dedup(tmp_path):
    rag = _pipeline(tmp_path)
    n = len(rag.chunks)
    assert rag.ingest_file("company_policy.md", (DOCS / "company_policy.md").read_bytes()) == 0
    rag2 = RAGPipeline(store_path=str(tmp_path / "idx.json"))
    assert len(rag2.chunks) == n
    assert "Northwind" in rag2.retrieve("annual leave carry over")[0].chunk.title


def test_incremental_ingest_avoids_full_rebuild():
    rag = _pipeline()
    rebuilds = rag.full_rebuilds
    n = len(rag.chunks)
    rag.ingest_text("Cafeteria", "The cafeteria serves vegetarian lunch every weekday from noon until two o'clock.")
    assert len(rag.chunks) == n + 1 and rag.full_rebuilds == rebuilds   # folded in, no refit
    assert len(rag.hnsw) == len(rag.chunks)
    assert rag.retrieve("when is vegetarian lunch served in the cafeteria")[0].chunk.title == "Cafeteria"
    for i in range(10):  # large growth triggers a refit
        rag.ingest_text(f"Extra {i}", f"Extra document number {i} about topic{i} and subject{i} matters.")
    assert rag.full_rebuilds > rebuilds


def test_delete_document(tmp_path):
    rag = _pipeline(tmp_path)
    docs = {d["title"]: d for d in rag.list_documents()}
    policy = next(d for t, d in docs.items() if "Northwind" in t)
    assert rag.delete_document(policy["doc_id"]) == policy["chunks"]
    assert all("Northwind" not in h.chunk.title for h in rag.retrieve("hotel reimbursement sick days", 5))
    assert len(rag.hnsw) == len(rag.chunks) == len(rag.bm25)
    assert len(RAGPipeline(store_path=str(tmp_path / "idx.json")).chunks) == len(rag.chunks)  # persisted
    assert rag.delete_document("nope") == 0


def test_grounding_check_and_citation_fix():
    rag = _pipeline()
    q = "How much is the hotel reimbursement per night?"
    hits = rag.retrieve(q, 3)
    good = "Hotel costs are reimbursed up to 180 euros per night in major cities and 130 euros elsewhere. [3]"
    g = rag.ground_answer(q, good, hits)
    assert g["grounded"] and g["answer"].endswith("[1]")          # wrong citation corrected
    off_topic = next(s for h in hits for s in h.chunk.text.split(". ") if "laptop" in s or "probation" in s)
    g = rag.ground_answer(q, off_topic + ".", hits)
    assert g["supported"] and not g["relevant"] and not g["grounded"]  # copied, but wrong sentence
    g = rag.ground_answer(q, "The moon is made of cheese and rockets.", hits)
    assert not g["supported"] and not g["grounded"]                  # hallucination
