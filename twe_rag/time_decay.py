# twe_rag/time_decay.py
from dataclasses import dataclass
import re
from datetime import datetime, timezone
from dateutil.parser import isoparse

RECENCY_TERMS = [
    r"current", r"latest", r"today", r"now", r"this year", r"who is", r"new", r"recent",
    r"as of", r"price", r"schedule", r"deadline", r"update"
]
REC_RE = re.compile(r"|".join(RECENCY_TERMS), re.IGNORECASE)

@dataclass
class DecayParams:
    delta: float  # weight for decay term
    tau_days: float

class TimeDecay:
    def __init__(self, base_delta=2.5, min_tau=90.0, max_tau=730.0):
        """
        Time decay with improved defaults for practical use.

        Args:
            base_delta: Base weight for decay term (default 2.5, up from 0.6)
                       Higher values make recency more important vs retrieval quality
            min_tau: Minimum tau in days for high-recency queries (default 90, up from 30)
                    With tau=90, e^(-90/90)=e^-1≈0.37 (37% weight at 3 months)
            max_tau: Maximum tau in days for low-recency queries (default 730, down from 3650)
                    With tau=730, e^(-365/730)=e^-0.5≈0.61 (61% weight at 1 year)
        """
        self.base_delta = base_delta
        self.min_tau = min_tau
        self.max_tau = max_tau

    def recency_need(self, query: str) -> float:
        """Detect recency need from query keywords. Returns 1.0 for recency queries, 0.3 otherwise."""
        return 1.0 if REC_RE.search(query) else 0.3  # increased from 0.2 to 0.3

    def params_for_query(self, query: str) -> DecayParams:
        need = self.recency_need(query)
        # Map: need=0.3 -> delta≈0.75, tau=638; need=1.0 -> delta=2.5, tau=90
        delta = self.base_delta * need
        tau = self.max_tau - (self.max_tau - self.min_tau) * need
        return DecayParams(delta=delta, tau_days=tau)

    def decay_value(self, doc_timestamp_iso: str, now: datetime, tau_days: float) -> float:
        try:
            dt = isoparse(doc_timestamp_iso)
        except Exception:
            dt = now
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (now - dt).total_seconds() / 86400.0
        import math
        return float(math.exp(-age_days / max(tau_days, 1e-3)))
