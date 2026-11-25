"""
HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index, cosine
similarity, pure numpy. askme delegates HNSW to Milvus/Weaviate/Qdrant; this is a compact
standalone implementation of Malkov & Yashunin (2016) so the whole stack runs in one process.

  * every node gets a random max level ~ floor(-ln U * mL), mL = 1/ln(M)
  * upper layers are sparse "express lanes", layer 0 holds every node
  * insert: greedy descent to the node's level, then beam search (ef_construction) per layer
    and connect to neighbours chosen by the diversity heuristic (Alg. 4)
  * search: greedy descent, then beam search with ef_search on layer 0
"""
from __future__ import annotations

import heapq
import math
import random

import numpy as np


class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ef_search: int = 64, seed: int = 0):
        self.dim, self.M, self.M0 = dim, M, 2 * M
        self.ef_construction, self.ef_search = ef_construction, ef_search
        self.mL = 1 / math.log(M)
        self.rng = random.Random(seed)
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.graph: list[dict[int, list[int]]] = []   # graph[level][node] -> neighbours
        self.entry: int | None = None
        self.max_level = -1

    def __len__(self):
        return len(self.vectors)

    # --------------------------------------------------------------------- helpers
    def _sim(self, q: np.ndarray, ids) -> np.ndarray:
        return self.vectors[ids] @ q

    def _search_layer(self, q, entry_points: list[int], ef: int, level: int) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns up to ef (sim, id) sorted best-first."""
        visited = set(entry_points)
        sims = self._sim(q, entry_points)
        cand = [(-s, e) for s, e in zip(sims, entry_points)]      # max-heap on sim
        best = [(s, e) for s, e in zip(sims, entry_points)]       # min-heap of current top-ef
        heapq.heapify(cand)
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)
        layer = self.graph[level]
        while cand:
            neg_s, c = heapq.heappop(cand)
            if -neg_s < best[0][0] and len(best) >= ef:
                break
            nbrs = [n for n in layer.get(c, ()) if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            for s, n in zip(self._sim(q, nbrs), nbrs):
                if len(best) < ef or s > best[0][0]:
                    heapq.heappush(cand, (-s, n))
                    heapq.heappush(best, (s, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted(best, reverse=True)

    def _select_neighbors(self, cands: list[tuple[float, int]], m: int) -> list[int]:
        """Diversity heuristic: keep a candidate only if it is closer to the base than to any kept one."""
        kept: list[int] = []
        for s, c in cands:  # cands sorted best-first
            if len(kept) >= m:
                break
            if all(float(self.vectors[c] @ self.vectors[k]) < s for k in kept):
                kept.append(c)
        if len(kept) < m:  # top up with the closest remaining ones
            for _, c in cands:
                if c not in kept:
                    kept.append(c)
                    if len(kept) >= m:
                        break
        return kept

    # --------------------------------------------------------------------- build
    def add(self, vecs: np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        start = len(self.vectors)
        self.vectors = np.concatenate([self.vectors, vecs])
        for i in range(start, len(self.vectors)):
            self._insert(i)

    def _insert(self, node: int):
        level = int(-math.log(1 - self.rng.random()) * self.mL)
        while len(self.graph) <= level:
            self.graph.append({})
        for lv in range(level + 1):
            self.graph[lv][node] = []
        if self.entry is None:
            self.entry, self.max_level = node, level
            return
        q = self.vectors[node]
        ep = [self.entry]
        for lv in range(self.max_level, level, -1):
            ep = [self._search_layer(q, ep, 1, lv)[0][1]]
        for lv in range(min(level, self.max_level), -1, -1):
            cands = self._search_layer(q, ep, self.ef_construction, lv)
            m_max = self.M0 if lv == 0 else self.M
            nbrs = self._select_neighbors(cands, self.M)
            self.graph[lv][node] = nbrs
            for n in nbrs:
                adj = self.graph[lv][n]
                adj.append(node)
                if len(adj) > m_max:
                    sims = self._sim(self.vectors[n], adj)
                    order = sorted(zip(sims, adj), reverse=True)
                    self.graph[lv][n] = self._select_neighbors(order, m_max)
            ep = [c for _, c in cands]
        if level > self.max_level:
            self.entry, self.max_level = node, level

    # --------------------------------------------------------------------- query
    def search(self, q: np.ndarray, k: int = 10, ef: int | None = None) -> list[tuple[int, float]]:
        if self.entry is None:
            return []
        q = np.asarray(q, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        ep = [self.entry]
        for lv in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, lv)[0][1]]
        res = self._search_layer(q, ep, max(ef or self.ef_search, k), 0)
        return [(i, float(s)) for s, i in res[:k]]
