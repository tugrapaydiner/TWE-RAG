# twe_rag/text_utils.py
import re
from typing import List, Set

_word = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return _word.findall(text.lower())

def shingles(tokens: List[str], n: int = 3) -> Set[str]:
    if len(tokens) < n:
        return set([' '.join(tokens)]) if tokens else set()
    return { ' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1) }
