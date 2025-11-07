# twe_rag/retrieval.py
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from joblib import load
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from twe_rag.types import Document
from twe_rag.text_utils import tokenize

IDX = Path('index')

class HybridRetriever:
    def __init__(self):
        # Check if indices exist
        required_files = [
            IDX/'bm25.joblib',
            IDX/'tfidf.joblib',
            IDX/'svd.joblib',
            IDX/'tfidf_svd.npy',
            IDX/'meta.json'
        ]

        missing = [f for f in required_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Index files not found: {[str(f) for f in missing]}\n\n"
                "Please run setup first:\n"
                "  python setup.py\n\n"
                "Or build indices manually:\n"
                "  python scripts/01_build_indices.py --svd-dim 128"
            )

        self.bm25: BM25Okapi = load(IDX/'bm25.joblib')
        self.tfidf: TfidfVectorizer = load(IDX/'tfidf.joblib')
        self.svd: TruncatedSVD = load(IDX/'svd.joblib')
        self.Xs = np.load(IDX/'tfidf_svd.npy')  # (N,d)
        meta = json.loads((IDX/'meta.json').read_text(encoding='utf-8'))
        self.ids = meta['ids']
        self.times = meta['timestamps']

    def _dense_embed(self, text: str) -> np.ndarray:
        vec = self.tfidf.transform([text])  # (1, V)
        sv = self.svd.transform(vec)        # (1, d)
        return normalize(sv)[0]             # (d,)

    def retrieve(self, query: str, K: int = 100, alpha: float = 1.0, beta: float = 1.0) -> List[Dict]:
        # Sparse scores
        q_tok = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tok)  # (N,)
        # Dense scores (cosine)
        qv = self._dense_embed(query)              # (d,)
        dv = self.Xs / (np.linalg.norm(self.Xs, axis=1, keepdims=True) + 1e-9)
        dense_scores = dv @ qv                     # (N,)

        # Combine (pre-normalize to comparable ranges)
        b = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-9)
        d = (dense_scores - dense_scores.min()) / (dense_scores.ptp() + 1e-9)
        combo = alpha*b + beta*d
        # Handle case where K > number of documents
        K_actual = min(K, len(combo))
        top = np.argpartition(-combo, K_actual-1)[:K_actual]
        top = top[np.argsort(-combo[top])]

        results = []
        for i in top:
            results.append({
                'doc': Document(id=self.ids[i], text=None, timestamp=self.times[i]),
                'idx': int(i),
                'partial': {'bm25': float(b[i]), 'dense': float(d[i])},
                'combo': float(combo[i])
            })
        return results
