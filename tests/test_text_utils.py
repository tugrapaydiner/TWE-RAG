from twe_rag.text_utils import tokenize, shingles

def test_tokenize_basic():
    assert tokenize('Hello, World!') == ['hello', 'world']

def test_shingles():
    s = shingles(['a','b','c','d'], n=3)
    assert 'a b c' in s and 'b c d' in s
