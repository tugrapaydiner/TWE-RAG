from twe_rag.budget import BudgetHalting

def test_halt_simple():
    bh = BudgetHalting(margin_thresh=0.1, agree_thresh=0.0)
    dec = bh.decide([0.9, 0.2], ["a b c", "a b d"])
    assert dec.halt
