# scripts/00_prepare_corpus.py
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
from dateutil.parser import isoparse

OUT = Path('data/corpus.jsonl')

def guess_timestamp(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.date().isoformat()

def from_folder(folder: Path):
    for p in folder.rglob('*'):
        if p.suffix.lower() in {'.txt', '.md'} and p.is_file():
            text = p.read_text(encoding='utf-8', errors='ignore')
            yield {
                'id': p.stem,
                'text': text,
                'timestamp': guess_timestamp(p)
            }

def from_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # validate
            assert 'id' in obj and 'text' in obj and 'timestamp' in obj
            # ensure ISO
            _ = isoparse(obj['timestamp'])
            yield obj

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--folder', type=Path)
    src.add_argument('--jsonl', type=Path)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open('w', encoding='utf-8') as w:
        if args.folder:
            for d in from_folder(args.folder):
                w.write(json.dumps(d, ensure_ascii=False) + '\n'); n += 1
        else:
            for d in from_jsonl(args.jsonl):
                w.write(json.dumps(d, ensure_ascii=False) + '\n'); n += 1
    print(f'Wrote {n} docs to {OUT}')
