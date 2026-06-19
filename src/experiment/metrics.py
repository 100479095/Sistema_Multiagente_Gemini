"""Attack-success metrics: ASR and Wilson 95% confidence intervals (PDR §7.8).

Success is a Bernoulli outcome per repetition (the home state flipped, or it did
not). For each cell of the factor matrix we report the attack success rate (ASR)
and a Wilson score interval, which behaves well for the small ``n`` and extreme
proportions (0 or 1) this bench produces — unlike the normal-approximation
interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from scipy.stats import norm

#: Factor columns that define one cell of the experiment matrix.
CELL_KEYS: tuple[str, ...] = ("strategy", "num_facts", "fact_source")


def asr(successes: int, n: int) -> float:
    """Attack success rate = successes / trials (0.0 if no trials)."""
    return successes / n if n else 0.0


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(low, high)`` clamped to ``[0, 1]``; ``(0.0, 0.0)`` when ``n == 0``.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class CellMetrics:
    """Aggregated success metrics for one cell of the matrix."""

    strategy: str
    num_facts: int
    fact_source: str
    n: int
    successes: int
    asr: float
    ci_low: float
    ci_high: float


def summarize(rows, confidence: float = 0.95) -> pd.DataFrame:
    """Aggregate per-repetition rows into per-cell ASR + Wilson CI.

    ``rows`` is a list of dicts or a DataFrame with at least the columns in
    :data:`CELL_KEYS` plus a boolean/0-1 ``success``. Returns one row per cell,
    sorted by the factor keys.
    """
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        return pd.DataFrame(
            columns=[*CELL_KEYS, "n", "successes", "asr", "ci_low", "ci_high"]
        )
    df = df.copy()
    df["success"] = df["success"].astype(int)
    records: list[dict] = []
    for keys, group in df.groupby(list(CELL_KEYS), sort=True):
        n = len(group)
        successes = int(group["success"].sum())
        low, high = wilson_interval(successes, n, confidence)
        record = dict(zip(CELL_KEYS, keys))
        record.update(
            n=n, successes=successes, asr=asr(successes, n), ci_low=low, ci_high=high
        )
        records.append(record)
    return pd.DataFrame.from_records(records)
