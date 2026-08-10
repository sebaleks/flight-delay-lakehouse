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


# --- Borderless "floating row" lists -----------------------------------------
# st.dataframe renders a spreadsheet: grid lines, an index gutter, and a fixed
# viewport that pads out with EMPTY ROWS when the data is shorter than the
# height. For a list you are meant to read (or pick from), that is the wrong
# object — it looks like a export, and the blank filler makes five results look
# like a broken table. These helpers render exactly as many rows as there are.

_ROW_CSS = """
<style>
.fdl-list { margin: .25rem 0 .5rem 0; }
.fdl-row {
  display: flex; align-items: center; gap: 1rem;
  padding: .55rem .75rem; border-radius: .5rem;
  border: none; box-shadow: none;
  transition: background-color .12s ease;
}
.fdl-row + .fdl-row { margin-top: .15rem; }
.fdl-row:hover { background: rgba(128,128,128,.10); }
.fdl-row .fdl-cell { flex: 1 1 0; min-width: 0; overflow: hidden; text-overflow: ellipsis;
                     white-space: nowrap; font-size: .93rem; }
.fdl-row .fdl-key { font-weight: 600; flex: 0 0 auto; min-width: 5.5rem; }
.fdl-row .fdl-num { text-align: right; font-variant-numeric: tabular-nums; }
.fdl-head { display: flex; gap: 1rem; padding: 0 .75rem .3rem .75rem;
            font-size: .78rem; letter-spacing: .04em; text-transform: uppercase;
            opacity: .6; }
.fdl-head .fdl-cell { flex: 1 1 0; }
.fdl-head .fdl-key { flex: 0 0 auto; min-width: 5.5rem; }
.fdl-head .fdl-num { text-align: right; }
</style>
"""


def inject_row_css() -> None:
    """Once per page, before any floating-row list."""
    if not st.session_state.get("_fdl_row_css"):
        st.markdown(_ROW_CSS, unsafe_allow_html=True)
        st.session_state["_fdl_row_css"] = True


def _esc(v: object) -> str:
    return (
        str(v)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def floating_rows(rows: list[dict], columns: list[tuple[str, str, str]]) -> None:
    """A borderless list, exactly len(rows) tall.

    columns is (field, header, kind) where kind is 'key' (bold, fixed width),
    'text', or 'num' (right-aligned, tabular figures). Values are HTML-escaped.
    """
    inject_row_css()
    head = "".join(f'<div class="fdl-cell fdl-{k}">{_esc(h)}</div>' for _, h, k in columns)
    body = []
    for r in rows:
        cells = "".join(
            f'<div class="fdl-cell fdl-{k}">{_esc(r.get(f, ""))}</div>' for f, _, k in columns
        )
        body.append(f'<div class="fdl-row">{cells}</div>')
    st.markdown(
        f'<div class="fdl-head">{head}</div><div class="fdl-list">{"".join(body)}</div>',
        unsafe_allow_html=True,
    )


def table_height(n_rows: int, row_px: int = 35, header_px: int = 38, max_px: int = 560) -> int:
    """Exact height for a st.dataframe with n_rows — no empty filler rows.

    Streamlit's default viewport pads short tables out with blanks, which reads
    as missing data. Capped so a long table still scrolls instead of running off
    the page.
    """
    return min(max(n_rows, 1) * row_px + header_px, max_px)
