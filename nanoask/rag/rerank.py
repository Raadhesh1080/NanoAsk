"""
Second-stage reranking (askme/rerank/rerank_service.py).

askme reranks with the BGE-reranker-v2-m3 cross-encoder. `CrossEncoderReranker` does the
same when `sentence-transformers` + a local model are available (`--reranker ce:<model>`).
The offline default, `FeatureReranker`, scores each (query, passage) *pair* jointly - the
property that makes a reranker better than first-stage retrieval - from:
  * idf-weighted query-term coverage   (did the passage mention what was asked?)
  * term proximity                     (smallest window that covers the matched terms)
  * ordered bigram / phrase matches
  * dense cosine similarity            (semantic match from the embedder)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nanoask.rag.bm25 import tokenize


@dataclass
class RerankCandidate:
    idx: int
    text: str
    dense_sim: float = 0.0
    first_stage: float = 0.0


class FeatureReranker:
    name = "feature"

    def __init__(self, idf_fn, w_cov=0.45, w_prox=0.15, w_phrase=0.15, w_dense=0.25):
        self.idf = idf_fn
        self.w = (w_cov, w_prox, w_phrase, w_dense)

    def score(self, query: str, cands: list[RerankCandidate]) -> list[float]:
        q = list(dict.fromkeys(tokenize(query)))
        q_idf = {t: self.idf(t) for t in q}
        total_idf = sum(q_idf.values()) or 1.0
        q_bigrams = set(zip(q, q[1:]))
        out = []
        for c in cands:
            toks = tokenize(c.text)
            present = {t for t in q if t in set(toks)}
            cov = sum(q_idf[t] for t in present) / total_idf
            prox = self._proximity(toks, present) if len(present) > 1 else (1.0 if present else 0.0)
            bigr = set(zip(toks, toks[1:]))
            phrase = len(q_bigrams & bigr) / len(q_bigrams) if q_bigrams else 0.0
            w_cov, w_prox, w_phrase, w_dense = self.w
            out.append(w_cov * cov + w_prox * prox + w_phrase * phrase + w_dense * max(c.dense_sim, 0.0))
        return out

    @staticmethod
    def _proximity(toks: list[str], terms: set[str]) -> float:
        """1 / (1 + log(span)) for the minimal window containing all present terms."""
        need, have, best, left = len(terms), {}, math.inf, 0
        for right, t in enumerate(toks):
            if t in terms:
                have[t] = have.get(t, 0) + 1
            while len(have) == need:
                best = min(best, right - left + 1)
                lt = toks[left]
                if lt in have:
                    have[lt] -= 1
                    if have[lt] == 0:
                        del have[lt]
                left += 1
        if best is math.inf:
            return 0.0
        return 1.0 / (1.0 + math.log(max(best - need + 1, 1)))


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.name = f"ce:{model_name}"

    def score(self, query: str, cands: list[RerankCandidate]) -> list[float]:
        return [float(s) for s in np.asarray(self.model.predict([(query, c.text) for c in cands]))]


def make_reranker(spec: str, idf_fn):
    if spec.startswith("ce:"):
        return CrossEncoderReranker(spec[3:])
    if spec == "none":
        return None
    return FeatureReranker(idf_fn)
