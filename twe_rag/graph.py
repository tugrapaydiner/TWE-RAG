# twe_rag/graph.py
from typing import List, Dict
import numpy as np
import networkx as nx

from twe_rag.text_utils import tokenize, shingles

class EvidenceGraph:
    def __init__(self, docs_texts: List[str]):
        self.docs_texts = docs_texts
        self._tokens = [tokenize(t) for t in docs_texts]
        self._shingles = [shingles(tok, n=3) for tok in self._tokens]

    def jaccard(self, i: int, j: int) -> float:
        a, b = self._shingles[i], self._shingles[j]
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b) + 1e-9
        return inter / union

    def degree_centrality(self, threshold: float = 0.05):
        n = len(self.docs_texts)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                w = self.jaccard(i, j)
                if w >= threshold:
                    G.add_edge(i, j, weight=w)
        deg = np.array([val for (_, val) in G.degree(weight='weight')], dtype=float)
        if deg.size == 0:
            return np.zeros(n, dtype=float)
        # normalize 0..1
        return (deg - deg.min()) / (deg.ptp() + 1e-9)

    def pagerank(self, threshold: float = 0.05, alpha: float = 0.85):
        n = len(self.docs_texts)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                w = self.jaccard(i, j)
                if w >= threshold:
                    G.add_edge(i, j, weight=w)
        if G.number_of_edges() == 0:
            return np.zeros(n, dtype=float)
        pr = nx.pagerank(G, alpha=alpha, weight='weight')
        vec = np.array([pr.get(i, 0.0) for i in range(n)], dtype=float)
        return (vec - vec.min()) / (vec.ptp() + 1e-9)
