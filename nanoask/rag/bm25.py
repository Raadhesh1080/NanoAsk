"""Okapi BM25 sparse retrieval (askme uses rank-bm25 / Milvus BM25; this is a self-contained version)."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOPWORDS = set("""a an and are as at be but by for from has have he her his i if in into is it its
of on or our she that the their them then there these they this to was we were what when where which
who why will with you your do does did can could would should how about than so not no yes""".split())

_WORD = re.compile(r"[\w]+", re.UNICODE)


def _stem(w: str) -> str:
    # deliberately tiny suffix stripper: enough to match "caches"/"cache", "batching"/"batch"
    for suf in ("ing", "edly", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.df: Counter = Counter()

    def __len__(self):
        return len(self.doc_len)

    def add(self, texts: list[str]):
        for text in texts:
            toks = tokenize(text)
            idx = len(self.doc_len)
            tf = Counter(toks)
            self.tf.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.df[t] += 1
                self.postings[t].append(idx)

    def idf(self, term: str) -> float:
        n, df = len(self.doc_len), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        if not self.doc_len:
            return []
        avgdl = sum(self.doc_len) / len(self.doc_len)
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf(term)
            for d in self.postings[term]:
                f = self.tf[d][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[d] / avgdl)
                scores[d] += idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
