"""Small shared formatting/display helpers reused across dashboard pages."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

# Fixed color language for delay severity, reused by every chart.
COLOR_DELAY = "#e4572e"  # warm red — "late"
COLOR_OK = "#2e86ab"  # calm blue — "on time"
COLOR_CANCEL = "#8338ec"  # violet — "cancelled"
COLOR_ACCENT = "#f4a261"
COLOR_TEXT = "#1b2733"  # matches .streamlit/config.toml textColor


def pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def minutes(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f} min"


def count(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{int(x):,}"


def download_button(df: pd.DataFrame, filename: str, label: str = "⬇ Download CSV") -> None:
    """Offer the current (filtered) table as a CSV — the curated gold layer,
    served as data to the consumer, not just pixels."""
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}",
    )
