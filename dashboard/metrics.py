"""Rate math for the additive views — the one rule that keeps numbers correct.

``dash_delays_by_time`` and ``dash_monthly_trend`` carry *additive counts*, never
pre-divided rates. Any rate at a rolled-up grain MUST be computed as
``SUM(numerator) / SUM(denominator)`` — never the mean of a per-row rate, which
would weight a 40-flight hour the same as a 40,000-flight one (dashboard_spec.md).

Every page that aggregates an additive view goes through here, so the invariant
lives in exactly one place (and Phase 5's correctness harness checks it).
"""

from __future__ import annotations

import pandas as pd

# label -> (numerator column, denominator column). Each ratio has its OWN
# denominator: departure delay is over flights that actually departed
# (n_with_dep_outcome), which differs from the arrival population because some
# flights depart and are then cancelled/diverted before arriving.
RATE_SPECS: dict[str, tuple[str, str]] = {
    "delay_rate": ("n_arr_del15", "n_with_arr_outcome"),
    "cancellation_rate": ("n_cancelled", "n_flights"),
    "diversion_rate": ("n_diverted", "n_flights"),
    "avg_arr_delay_minutes": ("sum_arr_delay_minutes", "n_with_arr_outcome"),
    "avg_dep_delay_minutes": ("sum_dep_delay_minutes", "n_with_dep_outcome"),
}

# The additive count/sum columns carried by the additive views.
ADDITIVE_COLS = [
    "n_flights",
    "n_with_arr_outcome",
    "n_with_dep_outcome",
    "n_arr_del15",
    "n_cancelled",
    "n_diverted",
    "sum_arr_delay_minutes",
    "sum_dep_delay_minutes",
]


def _with_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach every rate in RATE_SPECS to a frame that already holds summed
    additive columns. Zero denominators yield NaN, not a divide error."""
    out = frame.copy()
    for label, (num, den) in RATE_SPECS.items():
        denom = out[den].where(out[den] != 0)
        out[label] = out[num] / denom
    return out


def aggregate(df: pd.DataFrame, group_cols: list[str] | str) -> pd.DataFrame:
    """Group an additive view by ``group_cols``, SUM the counts, then derive
    rates from those sums. Returns one row per group with counts + rates."""
    keys = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    present = [c for c in ADDITIVE_COLS if c in df.columns]
    summed = df.groupby(keys, as_index=False, observed=True)[present].sum()
    return _with_rates(summed)


def totals(df: pd.DataFrame) -> dict[str, float]:
    """Collapse an entire additive view to overall scalar metrics
    (counts + SUM/SUM rates). Powers the scorecard row."""
    present = [c for c in ADDITIVE_COLS if c in df.columns]
    sums = df[present].sum()
    result: dict[str, float] = {c: float(sums[c]) for c in present}
    for label, (num, den) in RATE_SPECS.items():
        denom = sums[den]
        result[label] = float(sums[num] / denom) if denom else float("nan")
    return result
