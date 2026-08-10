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


def share_of(df: pd.DataFrame, col: str = "n_flights") -> pd.Series:
    """Each row's share of the column total. NaN-safe, sums to 1."""
    total = df[col].sum()
    return df[col] / total if total else df[col] * float("nan")


def mix_adjusted(
    df: pd.DataFrame,
    entity_col: str,
    stratum_col: str,
    *,
    num: str = "n_arr_del15",
    den: str = "n_with_arr_outcome",
) -> pd.DataFrame:
    """Each entity's rate against what its OWN mix of strata would predict.

    WHY A RAW RATE IS NOT A FAIR COMPARISON. Comparing airlines' overall delay
    rates across a set of routes silently compares their route mixes too: an
    airline concentrated on congested winter hubs looks worse than one flying
    Hawaii, whatever its operations are like. The question people actually mean
    is "is this airline worse than the routes it flies would predict?".

    So for each entity we compute the delays it WOULD have had performing at
    each stratum's own average, weighted by how much it flies there:

        expected = sum over strata of  n_entity_stratum * rate_stratum
        index    = actual / expected        (1.0 = exactly as its routes predict)

    This is indirect standardisation. It is only meaningful WITHIN the selected
    set — the strata rates are computed from the same filtered frame, so the
    index answers "relative to the other airlines here", not an absolute claim.

    Returns one row per entity with n, actual/expected counts, the raw rate, the
    expected rate, and the index.
    """
    strata = df.groupby(stratum_col, observed=True)[[num, den]].sum()
    strata_rate = (strata[num] / strata[den].where(strata[den] != 0)).rename("stratum_rate")
    joined = df.join(strata_rate, on=stratum_col)
    joined["expected"] = joined[den] * joined["stratum_rate"]

    out = joined.groupby(entity_col, observed=True).agg(
        n_flights=("n_flights", "sum"),
        actual=(num, "sum"),
        denominator=(den, "sum"),
        expected=("expected", "sum"),
    )
    out["rate"] = out["actual"] / out["denominator"].where(out["denominator"] != 0)
    out["expected_rate"] = out["expected"] / out["denominator"].where(out["denominator"] != 0)
    # index > 1 means worse than the mix of routes it flies would predict
    out["index"] = out["actual"] / out["expected"].where(out["expected"] != 0)
    return out.reset_index()
