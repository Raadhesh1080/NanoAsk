"""
Dense embedders.

askme uses BGE-M3 (dense + sparse). That needs a ~2 GB HuggingFace download, so the default
here is an offline Latent Semantic Analysis embedder (TF-IDF over stemmed unigrams+bigrams,
truncated SVD). It captures co-occurrence semantics ("KV cache" ~ "key/value memory") without
any download. If `sentence-transformers` and a local model are available, pass
`--embedder st:<model-name-or-path>` (e.g. BAAI/bge-m3) to use a real neural embedder.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from nanoask.rag.bm25 import tokenize


def _features(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


class LSAEmbedder:
    name = "lsa"
    refit_on_ingest = True  # the projection depends on the corpus

    def __init__(self, dim: int = 256):
        self.max_dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.proj: np.ndarray | None = None  # (V, d)
        self.dim = dim

    def _tfidf_rows(self, texts):
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            cnt = Counter(f for f in _features(t) if f in self.vocab)
            norm = math.sqrt(sum((1 + math.log(c)) ** 2 * self.idf[self.vocab[f]] ** 2 for f, c in cnt.items())) or 1.0
            for f, c in cnt.items():
                j = self.vocab[f]
                rows.append(r)
                cols.append(j)
                vals.append((1 + math.log(c)) * self.idf[j] / norm)
        return rows, cols, vals

    def fit(self, texts: list[str]):
        df = Counter()
        for t in texts:
            df.update(set(_features(t)))
        self.vocab = {f: i for i, f in enumerate(df)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[f])) + 1 for f in self.vocab], dtype=np.float32)
        rows, cols, vals = self._tfidf_rows(texts)
        A = torch.sparse_coo_tensor([rows, cols], vals, (n, len(self.vocab)), dtype=torch.float32, check_invariants=True)
        k = max(1, min(self.max_dim, n - 1, len(self.vocab) - 1))
        torch.manual_seed(0)
        _, S, V = torch.svd_lowrank(A, q=k, niter=6)
        self.proj = (V / S.clamp(min=1e-6)).numpy()  # fold-in: q_hat = q V S^-1
        self.dim = k

    def encode(self, texts: list[str]) -> np.ndarray:
        assert self.proj is not None, "call fit() first"
        rows, cols, vals = self._tfidf_rows(texts)
        M = np.zeros((len(texts), len(self.vocab)), dtype=np.float32) if len(self.vocab) < 200_000 else None
        if M is None:  # very large vocab: accumulate row by row
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            for r, c, v in zip(rows, cols, vals):
                out[r] += v * self.proj[c]
        else:
            M[rows, cols] = vals
            out = M @ self.proj
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


class SentenceTransformerEmbedder:
    refit_on_ingest = False

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, texts):
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def make_embedder(spec: str = "lsa"):
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    return LSAEmbedder()
