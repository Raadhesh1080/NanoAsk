"""Score fusion for hybrid search, ported from askme/retriever/base.py::SearchFusion."""
from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """score(d) = sum_i 1 / (k + rank_i(d)). Rank-based, so BM25 and cosine scales don't matter."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (doc, _) in enumerate(lst, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def alpha_fusion(dense: list[tuple[int, float]], sparse: list[tuple[int, float]], alpha: float = 0.5) -> list[tuple[int, float]]:
    """alpha * minmax(dense) + (1 - alpha) * minmax(sparse)."""
    def mm(lst):
        if not lst:
            return {}
        vals = [s for _, s in lst]
        lo, hi = min(vals), max(vals)
        return {d: (s - lo) / (hi - lo) if hi > lo else 1.0 for d, s in lst}
    nd, ns = mm(dense), mm(sparse)
    docs = set(nd) | set(ns)
    fused = {d: alpha * nd.get(d, 0.0) + (1 - alpha) * ns.get(d, 0.0) for d in docs}
    return sorted(fused.items(), key=lambda x: -x[1])
