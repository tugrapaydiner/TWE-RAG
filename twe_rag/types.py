# twe_rag/types.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Document:
    id: str
    text: str
    timestamp: str  # ISO date or datetime

@dataclass
class Retrieved:
    doc: Document
    score_parts: Dict[str, float]  # e.g., {"bm25":..., "dense":..., "centrality":..., "decay":...}
    score: float
