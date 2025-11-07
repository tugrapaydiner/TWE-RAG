# twe_rag/scoring.py
from typing import Dict

def combine_scores(parts: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(parts.get(k, 0.0) * weights.get(k, 0.0) for k in ['bm25','dense','centrality','decay'])
