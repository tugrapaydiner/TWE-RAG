# twe_rag/evals.py
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from twe_rag.pipeline import TWERAGPipeline, PipelineConfig

class Evaluator:
    def __init__(self, cfg: PipelineConfig):
        self.pipe = TWERAGPipeline(cfg)

    def exact_match(self, pred: str, gold: str) -> int:
        return int(pred.strip().lower() == gold.strip().lower())

    def run_toy_latest(self, qa_path: Path) -> Dict:
        n, correct = 0, 0
        for line in tqdm(qa_path.open('r', encoding='utf-8')):
            if not line.strip(): continue
            ex = json.loads(line)
            out = self.pipe.run(ex['question'])
            top = out['results'][0]['snippet'] if out['results'] else ''
            pred = ex['gold_latest'] if ex['gold_latest'].lower() in top.lower() else ''
            correct += self.exact_match(pred or '', ex['gold_latest'])
            n += 1
        return {'n': n, 'em': correct / max(n,1)}
