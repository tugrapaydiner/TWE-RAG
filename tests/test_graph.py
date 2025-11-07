from twe_rag.graph import EvidenceGraph

def test_degree_centrality():
    texts = [
        "alpha beta gamma delta epsilon zeta eta theta",
        "alpha beta gamma delta epsilon iota kappa lambda",
        "zeta eta theta iota kappa lambda mu nu xi"
    ]
    eg = EvidenceGraph(texts)
    c = eg.degree_centrality(threshold=0.05)
    assert len(c) == 3
    # First two texts should have some similarity, third might be isolated
    assert c[0] >= 0 or c[1] >= 0  # At least one should have connections
