"""
RAG pipeline (askme's ingest -> hybrid retrieve -> rerank -> generate flow).

    query ──► BM25 (sparse) ──┐
          └─► HNSW (dense) ───┴─► RRF / alpha fusion ─► top-N ─► rerank ─► top-k passages
                                                                               │
                                       prompt with numbered [n] citations ◄────┘
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from nanoask.rag.bm25 import BM25Index, tokenize
from nanoask.rag.documents import Chunk, ChunkingConfig, RecursiveChunker, load_bytes, make_chunks
from nanoask.rag.embeddings import make_embedder
from nanoask.rag.fusion import alpha_fusion, reciprocal_rank_fusion
from nanoask.rag.hnsw import HNSWIndex
from nanoask.rag.rerank import RerankCandidate, make_reranker

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the numbered context passages. "
    "Cite passages like [1]. If the answer is not in the context, say you don't know."
)


@dataclass
class RetrievalConfig:
    topk_candidates: int = 50     # per retriever
    rerank_top_n: int = 20        # fused candidates sent to the reranker
    final_k: int = 4              # passages put in the prompt
    fusion: str = "rrf"           # "rrf" | "alpha"
    rrf_k: int = 60
    alpha: float = 0.5            # weight of dense in alpha fusion


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float = 0.0

    def to_dict(self, rank: int):
        return {"rank": rank, "id": self.chunk.id, "title": self.chunk.title, "text": self.chunk.text,
                "score": round(self.score, 4), "fused_score": round(self.fused_score, 5),
                "bm25_rank": self.bm25_rank, "dense_rank": self.dense_rank,
                "source": self.chunk.metadata.get("source")}


class RAGPipeline:
    def __init__(self, embedder: str = "lsa", reranker: str = "feature",
                 chunking: ChunkingConfig | None = None, retrieval: RetrievalConfig | None = None,
                 store_path: str | None = None):
        self.chunker = RecursiveChunker(chunking)
        self.cfg = retrieval or RetrievalConfig()
        self.embedder = make_embedder(embedder)
        self.chunks: list[Chunk] = []
        self.bm25 = BM25Index()
        self.hnsw: HNSWIndex | None = None
        self.reranker = make_reranker(reranker, self.bm25.idf)
        self.store_path = Path(store_path) if store_path else None
        self._lock = threading.RLock()
        self._fitted_size = 0
        self.full_rebuilds = 0
        if self.store_path and self.store_path.exists():
            self._load()

    # ------------------------------------------------------------------ ingest
    def ingest_text(self, title: str, text: str, source: str = "") -> int:
        return self.add_chunks(make_chunks(title, text, self.chunker, source))

    def ingest_file(self, name: str, data: bytes) -> int:
        title, text = load_bytes(name, data)
        return self.ingest_text(title, text, source=name)

    def add_chunks(self, new: list[Chunk]) -> int:
        with self._lock:
            known = {c.id for c in self.chunks}
            new = [c for c in new if c.id not in known]   # re-uploading a doc is a no-op
            if not new:
                return 0
            self.chunks.extend(new)
            self.bm25.add([self._index_text(c) for c in new])
            self._rebuild_dense(new)
            self._save()
            return len(new)

    @staticmethod
    def _index_text(c: Chunk) -> str:
        return f"{c.title}\n{c.text}"

    # Refit the LSA projection only when the corpus has grown by this factor since the last fit;
    # in between, new chunks are "folded in" with the existing projection and appended to HNSW.
    REFIT_GROWTH = 1.25

    def _rebuild_dense(self, new: list[Chunk]):
        texts = [self._index_text(c) for c in self.chunks]
        needs_fit = self.hnsw is None or (
            self.embedder.refit_on_ingest and len(self.chunks) > self.REFIT_GROWTH * self._fitted_size)
        if needs_fit:
            self.embedder.fit(texts)
            self._fitted_size = len(self.chunks)
            self.hnsw = HNSWIndex(self.embedder.dim)
            self.hnsw.add(self.embedder.encode(texts))
            self.full_rebuilds += 1
        else:
            self.hnsw.add(self.embedder.encode([self._index_text(c) for c in new]))

    def _rebuild_all(self):
        self.bm25 = BM25Index()
        if self.reranker is not None and hasattr(self.reranker, "idf"):
            self.reranker.idf = self.bm25.idf
        self.hnsw, self._fitted_size = None, 0
        if self.chunks:
            self.bm25.add([self._index_text(c) for c in self.chunks])
            self._rebuild_dense(self.chunks)

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title,
                                           "source": c.metadata.get("source"), "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        """Removes a document. HNSW has no cheap delete, so the indexes are rebuilt (rare operation)."""
        with self._lock:
            before = len(self.chunks)
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
            removed = before - len(self.chunks)
            if removed:
                self._rebuild_all()
                self._save()
            return removed

    # ------------------------------------------------------------------ persistence
    def _save(self):
        if not self.store_path:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps([asdict(c) for c in self.chunks]))

    def _load(self):
        chunks = [Chunk(**d) for d in json.loads(self.store_path.read_text())]
        if chunks:
            self.chunks = chunks
            self.bm25.add([self._index_text(c) for c in chunks])
            self._rebuild_dense(chunks)

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        cfg = self.cfg
        k = k or cfg.final_k
        with self._lock:
            if not self.chunks:
                return []
            sparse = self.bm25.search(query, cfg.topk_candidates)
            qv = self.embedder.encode([query])[0]
            dense = self.hnsw.search(qv, cfg.topk_candidates)
            if cfg.fusion == "alpha":
                fused = alpha_fusion(dense, sparse, cfg.alpha)
            else:
                fused = reciprocal_rank_fusion([sparse, dense], cfg.rrf_k)
            bm_rank = {d: r for r, (d, _) in enumerate(sparse, 1)}
            de_rank = {d: r for r, (d, _) in enumerate(dense, 1)}
            de_sim = dict(dense)
            top = fused[: cfg.rerank_top_n]
            hits = [Hit(self.chunks[d], s, bm_rank.get(d), de_rank.get(d), s) for d, s in top]
            if self.reranker is not None and hits:
                cands = [RerankCandidate(d, self._index_text(self.chunks[d]),
                                         de_sim.get(d, float(self.hnsw.vectors[d] @ qv)), s) for d, s in top]
                for h, s in zip(hits, self.reranker.score(query, cands)):
                    h.score = s
                hits.sort(key=lambda h: -h.score)
            return hits[:k]

    # ------------------------------------------------------------------ generation helpers
    @staticmethod
    def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None,
                       max_context_chars: int = 6000) -> list[dict]:
        blocks, used = [], 0
        for i, h in enumerate(hits, 1):
            text = h.chunk.text
            if used + len(text) > max_context_chars:
                text = text[: max(0, max_context_chars - used)]
            if not text:
                break
            blocks.append(f"[{i}] ({h.chunk.title})\n{text}")
            used += len(text)
        context = "\n\n".join(blocks) if blocks else "(no relevant passages found)"
        user = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        return list(history or []) + [{"role": "user", "content": user}]

    @staticmethod
    def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
        """Model-free fallback: the best-matching sentences from the top passages, with citations."""
        if not hits:
            return "I couldn't find anything relevant in the uploaded documents."
        q = set(tokenize(question))
        scored = []
        for i, h in enumerate(hits, 1):
            for j, sent in enumerate(re.split(r"(?<=[.!?])\s+|\n+", h.chunk.text)):
                toks = tokenize(sent)
                if len(toks) < 4:
                    continue
                overlap = len(q & set(toks)) / (len(q) or 1)
                if overlap == 0:
                    continue
                scored.append((overlap + 0.1 / i - 0.01 * j, overlap, i, sent.strip()))
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1] if scored else 0
        # keep only sentences nearly as on-topic as the best one
        picked = [f"{s} [{i}]" for _, ov, i, s in scored[:max_sentences] if ov >= 0.5 * best]
        return " ".join(picked) or hits[0].chunk.text[:300] + " [1]"

    @staticmethod
    def ground_answer(question: str, answer: str, hits: list[Hit]) -> dict:
        """
        Post-generation faithfulness check (the generator may be a small model):
          * every answer sentence must be supported by some passage (>= 70% of its terms occur there);
            its citation is rewritten to the passage that actually supports it;
          * the supported content must be about the question: its question-term overlap must be at
            least half of the best sentence available in the retrieved passages.
        """
        q = set(tokenize(question))
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])|\n+", re.sub(r"\s*\[\d+\]", "", answer))
                 if len(tokenize(s)) >= 2]
        passages = [set(tokenize(h.chunk.text)) for h in hits]
        out, supported, rel = [], True, 0.0
        for s in sents:
            toks = set(tokenize(s))
            cover = [len(toks & p) / len(toks) for p in passages] or [0.0]
            best = max(range(len(cover)), key=cover.__getitem__)
            if cover[best] < 0.7:
                supported = False
                out.append(s)
                continue
            rel = max(rel, len(q & toks) / (len(q) or 1))
            out.append(f"{s} [{best + 1}]")
        best_avail = 0.0
        for h in hits:
            for s in re.split(r"(?<=[.!?])\s+|\n+", h.chunk.text):
                best_avail = max(best_avail, len(q & set(tokenize(s))) / (len(q) or 1))
        relevant = rel >= 0.5 * best_avail and rel > 0
        return {"answer": " ".join(out), "supported": bool(sents) and supported,
                "relevant": relevant, "grounded": bool(sents) and supported and relevant}

    def stats(self):
        return {"chunks": len(self.chunks), "documents": len({c.doc_id for c in self.chunks}),
                "embedder": self.embedder.name, "embedding_dim": self.embedder.dim,
                "reranker": getattr(self.reranker, "name", "none"), "fusion": self.cfg.fusion,
                "dense_full_rebuilds": self.full_rebuilds}
