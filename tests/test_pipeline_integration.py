import pytest
from pathlib import Path
from twe_rag.pipeline import TWERAGPipeline, PipelineConfig

def test_pipeline_runs():
    # This test will only pass if the corpus and indices are built
    if not Path('data/corpus.jsonl').exists() or not Path('index/bm25.joblib').exists():
        pytest.skip("Corpus or indices not built yet")

    pipe = TWERAGPipeline(PipelineConfig())
    out = pipe.run('current CEO of ExampleCorp')
    assert 'results' in out and 'meta' in out
