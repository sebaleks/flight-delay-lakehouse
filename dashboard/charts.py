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


def grouped_rate_col_v(
    df: pd.DataFrame,
    *,
    category: str,
    value_vars: dict[str, str],
    title: str,
    category_order: list | None = None,
    height: int = 380,
) -> go.Figure:
    """Vertical grouped columns: several rate columns per ordered category
    (e.g. delay rate + cancellation rate per season)."""
    long = df.melt(
        id_vars=[category],
        value_vars=list(value_vars),
        var_name="metric",
        value_name="rate",
    )
    long["metric"] = long["metric"].map(value_vars)
    fig = px.bar(
        long,
        x=category,
        y="rate",
        color="metric",
        barmode="group",
        title=title,
        color_discrete_sequence=[ui.COLOR_DELAY, ui.COLOR_CANCEL],
    )
    fig.update_yaxes(tickformat=".1%", title=None)
    fig.update_xaxes(title=None)
    if category_order is not None:
        fig.update_xaxes(categoryorder="array", categoryarray=category_order)
    fig.update_layout(legend_title_text=None, legend=dict(orientation="h", y=1.1, x=0))
    return _style(fig, height)


def rate_line_by_year(
    df: pd.DataFrame,
    *,
    x: str,
    rate_col: str,
    year_col: str,
    title: str,
    x_order: list | None = None,
    height: int = 420,
) -> go.Figure:
    """Overlaid year lines (one trace per year) of a rate across months."""
    d = df.copy()
    d[year_col] = d[year_col].astype(str)
    fig = px.line(
        d,
        x=x,
        y=rate_col,
        color=year_col,
        markers=True,
        title=title,
        color_discrete_sequence=px.colors.sequential.Oranges[3:][::-1],
    )
    fig.update_yaxes(tickformat=".0%", title="Delay rate")
    fig.update_xaxes(title=None)
    if x_order is not None:
        fig.update_xaxes(categoryorder="array", categoryarray=x_order)
    fig.update_layout(legend_title_text="Year", legend=dict(orientation="h", y=1.08, x=0))
    return _style(fig, height)


def rate_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 420,
) -> go.Figure:
    """Heatmap from a pre-pivoted rate matrix (index=rows, columns=x)."""
    hover = f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>delay rate: %{{z:.1%}}<extra></extra>"
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="OrRd",
            colorbar=dict(title="Delay rate", tickformat=".0%"),
            hovertemplate=hover,
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title=x_title, dtick=1)
    fig.update_yaxes(title=y_title, autorange="reversed")
    return _style(fig, height)


def airport_map(
    df: pd.DataFrame,
    *,
    lat: str,
    lon: str,
    size: str,
    color: str,
    hover_name: str,
    hover_cols: list[str],
    title: str,
    height: int = 560,
) -> go.Figure:
    """US bubble map — one marker per airport, sized by traffic, colored by
    delay rate. Uses scatter_geo (no mapbox token needed)."""
    fig = px.scatter_geo(
        df,
        lat=lat,
        lon=lon,
        size=size,
        color=color,
        hover_name=hover_name,
        hover_data=hover_cols,
        scope="usa",
        title=title,
        color_continuous_scale="OrRd",
        size_max=28,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white"), opacity=0.85))
    fig.update_geos(
        showland=True,
        landcolor="#f2f5f8",
        showlakes=False,
        subunitcolor="white",
        countrycolor="white",
    )
    fig.update_coloraxes(colorbar=dict(title="Delay rate", tickformat=".0%"))
    return _style(fig, height)


def scatter_volume_vs_rate(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hover_name: str,
    hover_cols: list[str],
    title: str,
    height: int = 460,
) -> go.Figure:
    """Traffic (log x) vs delay rate (y). Busy-and-late routes sit top-right."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_name=hover_name,
        hover_data=hover_cols,
        title=title,
        log_x=True,
        color=y,
        color_continuous_scale="OrRd",
        opacity=0.6,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_xaxes(title="Flight legs (log scale)")
    fig.update_yaxes(tickformat=".0%", title="Delay rate")
    fig.update_coloraxes(showscale=False)
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


def probability_vs_base_rate(
    p: float,
    *,
    base_rate: float,
    weather_known: bool = True,
    height: int = 150,
) -> go.Figure:
    """One flight's probability against the population rate.

    Deliberately NOT a gauge. A needle sitting in a red zone is a verdict
    wearing a probability costume; a bar on a 0-100% axis with the base rate
    marked lets the reader see BOTH how likely it is and how that compares,
    which is the only honest framing of a single calibrated number.

    weather_known=False desaturates the bar: an estimate made with all twelve
    weather features missing must not render with the same visual confidence as
    a weather-informed one.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[p],
            y=[""],
            orientation="h",
            marker_color=ui.COLOR_DELAY if weather_known else ui.COLOR_ACCENT,
            marker_opacity=1.0 if weather_known else 0.45,
            hovertemplate="modelled probability %{x:.0%}<extra></extra>",
            showlegend=False,
        )
    )
    # the base rate as a reference line + label, so the comparison is visual
    fig.add_vline(
        x=base_rate,
        line_width=2,
        line_dash="dot",
        line_color=ui.COLOR_TEXT,
        annotation_text=f"typical flight ({base_rate:.0%})",
        annotation_position="top",
        annotation_font_size=12,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title=None)
    fig.update_yaxes(showticklabels=False, title=None)
    fig.update_layout(bargap=0.45, showlegend=False)
    return _style(fig, height)


def reliability(
    bands: pd.DataFrame,
    *,
    highlight_lo: float | None = None,
    height: int = 340,
) -> go.Figure:
    """Predicted vs actual per band on the held-out set — the evidence panel.

    A perfectly calibrated model puts every bar on the diagonal. highlight_lo
    marks the band the user's own flight falls in, which is what turns a generic
    model-quality chart into an answer to "should I believe THIS number?".
    """
    mid = (bands["lo"] + bands["hi"]) / 2
    colors = [
        ui.COLOR_DELAY if (highlight_lo is not None and lo == highlight_lo) else ui.COLOR_OK
        for lo in bands["lo"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=mid,
            y=bands["frac_pos"],
            marker_color=colors,
            name="actually delayed",
            customdata=bands[["n"]].to_numpy(),
            hovertemplate=(
                "we said ~%{x:.0%}<br>actually delayed %{y:.1%}"
                "<br>%{customdata[0]:,} flights<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dot", color=ui.COLOR_TEXT, width=2),
            name="perfect calibration",
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="what the model said")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="what actually happened")
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    return _style(fig, height)


def ops_banks(
    banks: pd.DataFrame,
    *,
    title: str,
    height: int = 420,
) -> go.Figure:
    """Expected vs actual delayed departures per bank, with the model's own band.

    The band is ±2·sd where sd = √Σp(1−p) — the Poisson-binomial spread the
    calibrated probabilities themselves imply, assuming independence within
    the hour (shared shocks make the truth wider; the page says so). Actual
    counts overlay as points: inside the band the model called the bank,
    outside it something the features don't carry happened. An optional
    `baseline` column draws what history alone would have predicted for the
    same schedule.

    Columns: hour, expected, sd, actual, and optionally baseline.
    """
    hours = banks["hour"]
    upper = banks["expected"] + 2 * banks["sd"]
    lower = (banks["expected"] - 2 * banks["sd"]).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.concat([hours, hours[::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(46, 134, 171, 0.18)",  # COLOR_OK at low alpha
            line=dict(width=0),
            name="model band (±2sd)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=banks["expected"],
            mode="lines+markers",
            line=dict(color=ui.COLOR_OK, width=2),
            name="model expected (Σp)",
            hovertemplate="%{x}:00 — expected %{y:.1f}<extra></extra>",
        )
    )
    if "baseline" in banks.columns:
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=banks["baseline"],
                mode="lines",
                line=dict(color=ui.COLOR_TEXT, width=1.5, dash="dot"),
                name="history-implied",
                hovertemplate="%{x}:00 — history-implied %{y:.1f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=banks["actual"],
            mode="markers",
            marker=dict(color=ui.COLOR_DELAY, size=9, symbol="diamond"),
            name="actually delayed",
            hovertemplate="%{x}:00 — actual %{y}<extra></extra>",
        )
    )
    fig.update_xaxes(title="scheduled departure hour (local)", dtick=2)
    fig.update_yaxes(title="delayed departures (arr ≥ 15 min)", rangemode="tozero")
    fig.update_layout(title=title, legend=dict(orientation="h", y=1.1, x=0))
    return _style(fig, height)
