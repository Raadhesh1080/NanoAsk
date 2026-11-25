"""
HNSW recall on real chunk embeddings (not random vectors).

Index: all chunks of data/pypi_corpus + data/sample_docs, embedded with the same LSA embedder
the server uses. Queries: the 88 evaluation questions plus sentences sampled from chunks.
Ground truth: exact (brute-force) cosine top-k. Reports recall@k and latency for several
ef_search values, and the same for hnswlib (C++) as a reference if it is installed.

    python scripts/build_corpus.py      # once
    python scripts/eval_retrieval.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanoask.rag.documents import RecursiveChunker, load_path, make_chunks  # noqa: E402
from nanoask.rag.embeddings import LSAEmbedder  # noqa: E402
from nanoask.rag.hnsw import HNSWIndex  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=[str(ROOT / "data/pypi_corpus"), str(ROOT / "data/sample_docs")])
    ap.add_argument("--qa", default=str(ROOT / "data/eval/qa.jsonl"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sentence-queries", type=int, default=300)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(0)

    chunker = RecursiveChunker()
    chunks = [c for d in args.corpus for f in sorted(Path(d).glob("*")) if f.is_file()
              for c in make_chunks(*load_path(f), chunker, f.name)]
    texts = [f"{c.title}\n{c.text}" for c in chunks]
    emb = LSAEmbedder()
    emb.fit(texts)
    X = emb.encode(texts)

    queries = [json.loads(l)["q"] for l in open(args.qa) if l.strip()]
    sents = [s for t in texts for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 8]
    queries += rng.sample(sents, min(args.sentence_queries, len(sents)))
    Q = emb.encode(queries)
    truth = np.argsort(-(Q @ X.T), axis=1)[:, : args.k]
    print(f"{len(chunks)} chunks, dim {emb.dim}, {len(queries)} queries "
          f"({len(queries) - args.sentence_queries} questions + {args.sentence_queries} sentences), k={args.k}")

    def report(name, search, build_s):
        t = time.perf_counter()
        got = [search(q) for q in Q]
        ms = (time.perf_counter() - t) / len(Q) * 1000
        rec = np.mean([len(set(g) & set(tr)) / args.k for g, tr in zip(got, truth)])
        print(f"  {name:28s} recall@{args.k} {rec:6.3f}   {ms:6.2f} ms/query   build {build_s:5.1f}s")

    t = time.perf_counter()
    idx = HNSWIndex(emb.dim, M=args.M, ef_construction=args.ef_construction)
    idx.add(X)
    build = time.perf_counter() - t
    t = time.perf_counter()
    _ = Q @ X.T
    print(f"  {'brute force (numpy)':28s} recall@{args.k}  1.000   "
          f"{(time.perf_counter() - t) / len(Q) * 1000:6.2f} ms/query")
    for ef in (16, 32, 64, 128):
        report(f"nanoask HNSW ef={ef}", lambda q, ef=ef: [i for i, _ in idx.search(q, args.k, ef=ef)], build)
    try:
        import hnswlib
        t = time.perf_counter()
        h = hnswlib.Index(space="ip", dim=emb.dim)
        h.init_index(max_elements=len(X), M=args.M, ef_construction=args.ef_construction, random_seed=0)
        h.add_items(X)
        hb = time.perf_counter() - t
        for ef in (16, 64):
            h.set_ef(max(ef, args.k))
            report(f"hnswlib (C++) ef={ef}", lambda q: list(h.knn_query(q, k=args.k)[0][0]), hb)
    except ImportError:
        print("  (hnswlib not installed; skipping reference)")


if __name__ == "__main__":
    main()
