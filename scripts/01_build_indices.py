# scripts/01_build_indices.py
import json, argparse
from pathlib import Path
from tqdm import tqdm
from joblib import dump
from dateutil.parser import isoparse

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from twe_rag.text_utils import tokenize

DATA = Path('data/corpus.jsonl')
IDX = Path('index')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--svd-dim', type=int, default=128)
    args = ap.parse_args()

    IDX.mkdir(parents=True, exist_ok=True)

    docs, ids, times, tokenized = [], [], [], []
    with DATA.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # validate timestamp parses
            _ = isoparse(obj['timestamp'])
            ids.append(obj['id'])
            times.append(obj['timestamp'])
            text = obj['text']
            docs.append(text)
            tokenized.append(tokenize(text))

    # BM25
    bm25 = BM25Okapi(tokenized)
    dump(bm25, IDX/'bm25.joblib')

    # TF-IDF + SVD (dense-ish, 128D)
    tfidf = TfidfVectorizer(max_features=50000)
    X = tfidf.fit_transform(docs)
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=42)
    Xs = svd.fit_transform(X)  # (N, d)

    dump(tfidf, IDX/'tfidf.joblib')
    dump(svd, IDX/'svd.joblib')
    np.save(IDX/'tfidf_svd.npy', Xs)

    meta = { 'ids': ids, 'timestamps': times }
    (IDX/'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    print(f'Indexed {len(ids)} docs; SVD dim={args.svd_dim}')
