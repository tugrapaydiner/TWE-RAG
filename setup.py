#!/usr/bin/env python3
"""
One-command setup script for TWE-RAG.
Handles corpus generation, preparation, and indexing.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

print("=" * 60)
print("TWE-RAG Setup Script")
print("=" * 60)

# Step 1: Generate sample corpus
print("\n[1/3] Generating sample corpus...")
OUT_DIR = Path('data/raw_texts')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = {
    'product_launch': [
        'announced the release of {product} version {version}',
        'unveiled {product}, a new solution for {use_case}',
        'introduced {product} with enhanced {feature} capabilities',
    ],
    'financial': [
        'reported quarterly revenue of ${amount}M, {trend} {percent}%',
        'stock price {movement} to ${price} per share',
        'announced {type} investment of ${amount}M in {area}',
    ],
    'partnership': [
        'formed strategic partnership with {partner} to {goal}',
        'acquired {company} for ${amount}M to expand {area}',
        'announced collaboration with {partner} on {project}',
    ],
}

PRODUCTS = ['CloudSync', 'DataVault', 'AIAssist', 'SecureNet', 'DevTools', 'AnalyticsPro']
FEATURES = ['security', 'performance', 'scalability', 'reliability', 'automation']
USE_CASES = ['enterprise workflows', 'data analytics', 'cloud migration', 'DevOps']
PARTNERS = ['TechCorp', 'DataSystems Inc', 'CloudFirst', 'InnovateLabs']

def generate_document(doc_id: int, date: datetime) -> dict:
    topic = random.choice(list(TOPICS.keys()))
    template = random.choice(TOPICS[topic])

    content = template.format(
        product=random.choice(PRODUCTS),
        version=f'{random.randint(1,5)}.{random.randint(0,9)}',
        use_case=random.choice(USE_CASES),
        feature=random.choice(FEATURES),
        amount=random.randint(10, 500),
        percent=random.randint(5, 50),
        trend='up' if random.random() > 0.5 else 'down',
        movement='rose' if random.random() > 0.5 else 'fell',
        price=random.randint(50, 300),
        type='strategic' if random.random() > 0.5 else 'capital',
        area=random.choice(['R&D', 'infrastructure', 'talent', 'expansion']),
        partner=random.choice(PARTNERS),
        goal='enhance customer experience',
        company=random.choice(PARTNERS),
        project='next-generation platform',
    )

    title = f"ExampleCorp News - {date.strftime('%B %Y')}"
    text = f"""{title}

{date.strftime('%B %d, %Y')} - ExampleCorp {content}. This development represents a significant milestone for the company and reflects its ongoing commitment to innovation and excellence.

Industry analysts have responded positively to the announcement, noting ExampleCorp's strong position in the market. The company continues to invest in cutting-edge technologies and strategic initiatives that drive long-term value.

"This is an exciting time for ExampleCorp and our customers," said a company spokesperson. "We remain focused on delivering exceptional solutions that meet the evolving needs of the market."

ExampleCorp has been a leader in enterprise technology for over {random.randint(10, 25)} years, serving thousands of customers worldwide."""

    return {
        'id': f'doc_{doc_id:04d}',
        'text': text,
        'timestamp': date.date().isoformat()
    }

# Generate documents
start_date = datetime(2019, 1, 1)
docs = []
for i in range(60):
    days_offset = i * 36
    doc_date = start_date + timedelta(days=days_offset)
    doc = generate_document(i, doc_date)
    docs.append(doc)

# Add CEO documents
ceo_docs = [
    {
        'id': 'doc_ceo_2019',
        'text': 'ExampleCorp Appoints New CEO\n\nJanuary 15, 2019 - ExampleCorp announced that Alice Newton has been appointed as the new Chief Executive Officer. Alice Newton brings over 20 years of experience in the technology sector.',
        'timestamp': '2019-01-15'
    },
    {
        'id': 'doc_ceo_2022',
        'text': 'ExampleCorp Leadership Transition\n\nJune 1, 2022 - ExampleCorp announced a major leadership change today. Bob Ortega will take over as Chief Executive Officer, replacing Alice Newton who is stepping down after three successful years.',
        'timestamp': '2022-06-01'
    },
    {
        'id': 'doc_ceo_2024',
        'text': 'ExampleCorp Names Cara Singh as New CEO\n\nSeptember 10, 2024 - ExampleCorp announced today that Cara Singh has been appointed as the company\'s new Chief Executive Officer, effective immediately. She succeeds Bob Ortega.',
        'timestamp': '2024-09-10'
    }
]
docs.extend(ceo_docs)

print(f"Generated {len(docs)} documents")

# Step 2: Write corpus
print("\n[2/3] Writing corpus file...")
corpus_path = Path('data/corpus.jsonl')
corpus_path.parent.mkdir(parents=True, exist_ok=True)

with corpus_path.open('w', encoding='utf-8') as f:
    for doc in docs:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

print(f"Wrote {len(docs)} docs to {corpus_path}")

# Step 3: Build indices
print("\n[3/3] Building indices...")
sys.path.insert(0, str(Path(__file__).parent))

from dateutil.parser import isoparse
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from joblib import dump
from twe_rag.text_utils import tokenize

IDX = Path('index')
IDX.mkdir(parents=True, exist_ok=True)

ids, times, texts, tokenized = [], [], [], []
with corpus_path.open('r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        ids.append(obj['id'])
        times.append(obj['timestamp'])
        texts.append(obj['text'])
        tokenized.append(tokenize(obj['text']))

# BM25
bm25 = BM25Okapi(tokenized)
dump(bm25, IDX/'bm25.joblib')
print("  ✓ Built BM25 index")

# TF-IDF + SVD
tfidf = TfidfVectorizer(max_features=50000)
X = tfidf.fit_transform(texts)
svd = TruncatedSVD(n_components=128, random_state=42)
Xs = svd.fit_transform(X)

dump(tfidf, IDX/'tfidf.joblib')
dump(svd, IDX/'svd.joblib')
np.save(IDX/'tfidf_svd.npy', Xs)
print("  ✓ Built TF-IDF and SVD indices")

# Metadata
meta = {'ids': ids, 'timestamps': times}
(IDX/'meta.json').write_text(json.dumps(meta), encoding='utf-8')
print("  ✓ Saved metadata")

print(f"\n{'=' * 60}")
print("Setup Complete!")
print(f"{'=' * 60}")
print(f"\nIndexed {len(ids)} documents with 128D dense vectors")
print("\nNext steps:")
print("  1. Run a query:")
print('     python scripts/02_run_query.py --q "Who is the current CEO?"')
print("\n  2. Run tests:")
print("     python -m pytest tests/ -v")
print("\n  3. Launch demo:")
print("     streamlit run demo/app.py")
print()
