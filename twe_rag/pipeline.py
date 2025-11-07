# twe_rag/pipeline.py
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict

import numpy as np

from twe_rag.types import Retrieved, Document
from twe_rag.retrieval import HybridRetriever
from twe_rag.graph import EvidenceGraph
from twe_rag.time_decay import TimeDecay
from twe_rag.budget import BudgetHalting
from twe_rag.scoring import combine_scores
from twe_rag.io_utils import CorpusIO

@dataclass
class PipelineConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    # delta handled by TimeDecay per query
    K_stages: List[int] = None  # e.g., [30, 60, 100]
    # Time decay parameters (configurable for experimentation)
    base_delta: float = 2.5  # Weight for decay term
    min_tau: float = 90.0    # Min tau in days (for recency queries)
    max_tau: float = 730.0   # Max tau in days (for historical queries)

class TWERAGPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        if self.cfg.K_stages is None:
            self.cfg.K_stages = [30, 60, 100]
        self.ret = HybridRetriever()
        self.io = CorpusIO()
        self.decay = TimeDecay(
            base_delta=self.cfg.base_delta,
            min_tau=self.cfg.min_tau,
            max_tau=self.cfg.max_tau
        )
        self.halt = BudgetHalting()

    def run(self, query: str, now: datetime = None) -> Dict:
        now = now or datetime.now(timezone.utc)
        # get decay params from query
        dp = self.decay.params_for_query(query)

        best_stage = None
        stage_results: List[Retrieved] = []

        for K in self.cfg.K_stages:
            cand = self.ret.retrieve(query, K=K, alpha=self.cfg.alpha, beta=self.cfg.beta)
            # load texts for candidates
            texts = [self.io.get_text(c['doc'].id) if c['doc'].text is None else c['doc'].text for c in cand]
            # evidence graph centrality
            eg = EvidenceGraph(texts)
            central = eg.degree_centrality(threshold=0.05)
            # decay per doc
            decays = []
            for c in cand:
                ts = self.io.get_timestamp(c['doc'].id)
                decays.append(self.decay.decay_value(ts, now=now, tau_days=dp.tau_days))
            # final scores
            results: List[Retrieved] = []
            scores = []
            for i, c in enumerate(cand):
                parts = {
                    'bm25': c['partial']['bm25'],
                    'dense': c['partial']['dense'],
                    'centrality': float(central[i]),
                    'decay': float(decays[i]),
                }
                score = combine_scores(parts, weights={
                    'bm25': self.cfg.alpha,
                    'dense': self.cfg.beta,
                    'centrality': self.cfg.gamma,
                    'decay': dp.delta,
                })
                results.append(Retrieved(doc=Document(id=c['doc'].id, text=texts[i], timestamp=self.io.get_timestamp(c['doc'].id)),
                                         score_parts=parts, score=score))
                scores.append(score)

            # sort
            order = np.argsort([-r.score for r in results])
            results = [results[i] for i in order]
            scores = [results[i].score for i in range(min(len(results), 5))]
            top_texts = [results[i].doc.text for i in range(min(len(results), 5))]

            dec = self.halt.decide([r.score for r in results[:5]], top_texts)
            stage_results = results
            best_stage = {
                'K': K,
                'halted': dec.halt,
                'reason': dec.reason,
                'decay_params': dp.__dict__,
            }
            if dec.halt:
                break

        return {
            'query': query,
            'meta': best_stage,
            'results': [
                {
                    'id': r.doc.id,
                    'timestamp': r.doc.timestamp,
                    'score': r.score,
                    'parts': r.score_parts,
                    'snippet': r.doc.text[:400]
                } for r in stage_results[:10]
            ]
        }
