# experiments/run_eval.py
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from twe_rag.evals import Evaluator
from twe_rag.pipeline import PipelineConfig

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--qa', type=Path, default=Path('data/toy_qa.jsonl'))
    args = ap.parse_args()

    ev = Evaluator(PipelineConfig(alpha=1.0, beta=1.0, gamma=0.5))
    res = ev.run_toy_latest(args.qa)
    print(res)
