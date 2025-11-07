# twe_rag/budget.py
from dataclasses import dataclass
from typing import List
import numpy as np
from twe_rag.text_utils import tokenize, shingles

@dataclass
class HaltDecision:
    halt: bool
    reason: str

class BudgetHalting:
    def __init__(self, margin_thresh=0.15, agree_thresh=0.12, agree_k=5):
        self.margin_thresh = margin_thresh
        self.agree_thresh = agree_thresh
        self.agree_k = agree_k

    def agreement(self, texts: List[str]) -> float:
        k = min(self.agree_k, len(texts))
        if k < 2:
            return 0.0
        sets = [shingles(tokenize(t), n=3) for t in texts[:k]]
        sims = []
        for i in range(k):
            for j in range(i+1, k):
                a, b = sets[i], sets[j]
                if not a or not b:
                    sims.append(0.0); continue
                inter = len(a & b)
                union = len(a | b) + 1e-9
                sims.append(inter/union)
        return float(np.mean(sims)) if sims else 0.0

    def decide(self, top_scores: List[float], top_texts: List[str]) -> HaltDecision:
        if len(top_scores) < 2:
            return HaltDecision(halt=True, reason='single candidate')
        s = (np.array(top_scores) - min(top_scores)) / (max(top_scores) - min(top_scores) + 1e-9)
        margin = float(s[0] - s[1])
        agree = self.agreement(top_texts)
        if margin >= self.margin_thresh and agree >= self.agree_thresh:
            return HaltDecision(halt=True, reason=f'margin={margin:.3f}, agree={agree:.3f}')
        return HaltDecision(halt=False, reason=f'margin={margin:.3f}, agree={agree:.3f}')
