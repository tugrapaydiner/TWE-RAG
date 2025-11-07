# twe_rag/io_utils.py
import json
from pathlib import Path
from typing import Dict

DATA = Path('data/corpus.jsonl')

class CorpusIO:
    def __init__(self):
        self._by_id: Dict[str, Dict] = {}

        if not DATA.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {DATA}\n\n"
                "Please run setup first:\n"
                "  python setup.py\n\n"
                "Or prepare corpus manually:\n"
                "  python scripts/00_prepare_corpus.py --jsonl data/sample_corpus.jsonl"
            )

        with DATA.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self._by_id[obj['id']] = obj

        if not self._by_id:
            raise ValueError(
                f"Corpus file is empty: {DATA}\n\n"
                "Please run setup to generate sample data:\n"
                "  python setup.py"
            )

    def get_text(self, doc_id: str) -> str:
        return self._by_id[doc_id]['text']

    def get_timestamp(self, doc_id: str) -> str:
        return self._by_id[doc_id]['timestamp']
