"""Line chart visualizations (static matplotlib + interactive Plotly JSON)
showing per-model metric trends across attack categories.
"""
import json

import matplotlib.pyplot as plt
from pandas import DataFrame

from evaluation.report.config import (
    MODEL_COLORS,
    MODEL_LINESTYLES,
    MODEL_MARKERS,
    display_name,
)
from evaluation.visualizations.utils import extract_series


def line_chart(df: DataFrame, metric: str, group_col: str) -> plt.Figure:
    models, categories, series = extract_series(df, metric, group_col)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    for model in models:
        color = MODEL_COLORS.get(model, "#888888")
        marker = MODEL_MARKERS.get(model, "o")
        ls = MODEL_LINESTYLES.get(model, "-")
        ax.plot([display_name(c) for c in categories], series[model],
                marker=marker, linestyle=ls, linewidth=2, label=display_name(model),
                color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.2)
    ax.set_ylabel(display_name(metric), fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, handlelength=2.5)
    plt.xticks(rotation=25, ha="right", fontsize=9.5)
    fig.subplots_adjust(right=0.78)
    fig.tight_layout()
    return fig


_PLOTLY_DASH = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}


def line_plotly(df: DataFrame, metric: str, group_col: str) -> str:
    models, categories, series = extract_series(df, metric, group_col)
    d_cats = [display_name(c) for c in categories]

    traces = []
    for model in models:
        color = MODEL_COLORS.get(model, "#888888")
        dash = _PLOTLY_DASH.get(MODEL_LINESTYLES.get(model, "-"), "solid")
        traces.append({
            "type": "scatter", "mode": "lines+markers",
            "x": d_cats, "y": series[model],
            "name": display_name(model),
            "line": {"color": color, "width": 2.5, "dash": dash},
            "marker": {"size": 7},
            "hovertemplate": "%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
        })

    spec = {
        "data": traces,
        "layout": {
            "yaxis": {
                "title": display_name(metric),
                "range": [-0.05, 1.08],
                "tickformat": ".0%",
            },
            "xaxis": {"tickangle": -25},
            "font": {"family": "Inter, sans-serif", "size": 12},
            "showlegend": True,
            "legend": {"font": {"size": 10}, "x": 1.02, "y": 1},
            "margin": {"l": 60, "r": 160, "t": 30, "b": 80},
            "height": 450,
            "paper_bgcolor": "#fff", "plot_bgcolor": "#fff",
            "hovermode": "x unified",
        },
    }
    return json.dumps(spec)
