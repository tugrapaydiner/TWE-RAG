# scripts/02_run_query.py
import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from twe_rag.pipeline import TWERAGPipeline, PipelineConfig

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', required=True, help='query text')
    ap.add_argument('--alpha', type=float, default=1.0, help='BM25 weight')
    ap.add_argument('--beta', type=float, default=1.0, help='Dense vector weight')
    ap.add_argument('--gamma', type=float, default=0.5, help='Centrality weight')
    ap.add_argument('--stages', type=int, nargs='+', default=[30,60,100], help='K stages for budgeted retrieval')
    ap.add_argument('--base-delta', type=float, default=2.5, help='Time decay weight (higher = stronger recency)')
    ap.add_argument('--min-tau', type=float, default=90.0, help='Min tau in days for recency queries')
    ap.add_argument('--max-tau', type=float, default=730.0, help='Max tau in days for historical queries')
    args = ap.parse_args()

    pipe = TWERAGPipeline(PipelineConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        K_stages=args.stages,
        base_delta=args.base_delta,
        min_tau=args.min_tau,
        max_tau=args.max_tau
    ))
    out = pipe.run(args.q)
    print(json.dumps(out, ensure_ascii=False, indent=2))
