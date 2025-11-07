from datetime import datetime, timezone, timedelta
from twe_rag.time_decay import TimeDecay

def test_decay_monotonic():
    td = TimeDecay()
    now = datetime.now(timezone.utc)
    newer = (now - timedelta(days=1)).isoformat()
    older = (now - timedelta(days=365)).isoformat()
    p = td.params_for_query('latest news')
    assert td.decay_value(newer, now, p.tau_days) > td.decay_value(older, now, p.tau_days)
