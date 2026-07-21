"""Reusable Plotly chart builders with one consistent house style.

Every page draws through here so axes, hover, fonts, and the delay/cancel color
language stay identical across the dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard import ui

# One layout applied to every figure.
_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="sans-serif", size=13, color=ui.COLOR_TEXT),
    margin=dict(l=10, r=10, t=48, b=10),
    hoverlabel=dict(font_size=13),
    title=dict(font=dict(size=17)),
)


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, **_LAYOUT)
    return fig


def rate_bar_h(
    df: pd.DataFrame,
    *,
    category: str,
    rate_col: str,
    title: str,
    hover_cols: list[str] | None = None,
    height: int = 460,
    color: str | None = None,
) -> go.Figure:
    """Horizontal bar of a rate by category, worst (highest) at the top."""
    d = df.sort_values(rate_col, ascending=True)
    fig = px.bar(
        d,
        x=rate_col,
        y=category,
        orientation="h",
        title=title,
        hover_data=hover_cols or [],
    )
    fig.update_traces(marker_color=color or ui.COLOR_DELAY)
    fig.update_xaxes(tickformat=".0%", title=None)
    fig.update_yaxes(title=None)
    return _style(fig, height)


def rate_col_v(
    df: pd.DataFrame,
    *,
    category: str,
    rate_col: str,
    title: str,
    height: int = 380,
    color: str | None = None,
    category_order: list | None = None,
) -> go.Figure:
    """Vertical column chart of a rate by an ordered category."""
    fig = px.bar(df, x=category, y=rate_col, title=title)
    fig.update_traces(marker_color=color or ui.COLOR_DELAY)
    fig.update_yaxes(tickformat=".0%", title="Delay rate")
    fig.update_xaxes(title=None)
    if category_order is not None:
        fig.update_xaxes(categoryorder="array", categoryarray=category_order)
    return _style(fig, height)


def grouped_rate_bar_h(
    df: pd.DataFrame,
    *,
    category: str,
    value_vars: dict[str, str],
    title: str,
    height: int = 480,
) -> go.Figure:
    """Horizontal grouped bars: several rate columns per category
    (e.g. delay rate + cancellation rate per carrier)."""
    long = df.melt(
        id_vars=[category],
        value_vars=list(value_vars),
        var_name="metric",
        value_name="rate",
    )
    long["metric"] = long["metric"].map(value_vars)
    order = df.sort_values(next(iter(value_vars)), ascending=True)[category].tolist()
    fig = px.bar(
        long,
        x="rate",
        y=category,
        color="metric",
        orientation="h",
        barmode="group",
        title=title,
        color_discrete_sequence=[ui.COLOR_DELAY, ui.COLOR_CANCEL],
    )
    fig.update_xaxes(tickformat=".1%", title=None)
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=order)
    fig.update_layout(legend_title_text=None, legend=dict(orientation="h", y=1.08, x=0))
    return _style(fig, height)
