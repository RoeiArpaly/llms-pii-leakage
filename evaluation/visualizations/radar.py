"""Radar chart visualizations (static matplotlib + interactive Plotly JSON)
for comparing multi-dimensional model performance across attack categories.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from evaluation.report.config import (
    MODEL_COLORS,
    display_name,
)
from evaluation.visualizations.utils import extract_series


def radar_chart(df: DataFrame, metric: str, group_col: str) -> plt.Figure:
    models, categories, series = extract_series(df, metric, group_col)
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200, subplot_kw={"polar": True})
    ax.set_facecolor("#fafafa")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for model in models:
        values = series[model] + series[model][:1]
        color = MODEL_COLORS.get(model, "#888888")
        ax.plot(angles, values, "-", linewidth=1.8, label=display_name(model),
                color=color, alpha=0.9)
        ax.fill(angles, values, alpha=0.06, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([display_name(c) for c in categories], fontsize=7.5, color="#333333")
    ax.set_ylim(0, 1.05)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.5, color="#888888")
    ax.spines["polar"].set_color("#cccccc")
    ax.grid(color="#cccccc", linewidth=0.5)

    title = f"{display_name(metric)} — Models"
    ax.set_title(title, fontsize=9, fontweight="bold", pad=18)

    ax.legend(
        loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=7,
        ncol=1, handlelength=1.8, frameon=True, edgecolor="#ccc",
    )
    fig.tight_layout()
    return fig


def radar_plotly(df: DataFrame, metric: str, group_col: str) -> str:
    models, categories, series = extract_series(df, metric, group_col)
    d_cats = [display_name(c) for c in categories]

    title = f"{display_name(metric)} — Models"

    def _rgba(hex_color, alpha):
        h = hex_color.lstrip("#")
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    traces = []
    for model in models:
        vals = series[model] + series[model][:1]
        theta = d_cats + d_cats[:1]
        color = MODEL_COLORS.get(model, "#888888")
        traces.append({
            "type": "scatterpolar", "r": vals, "theta": theta,
            "fill": "toself",
            "fillcolor": _rgba(color, 0.18),
            "line": {"color": color, "width": 2},
            "name": display_name(model),
            "hovertemplate": "%{theta}<br>%{r:.1%}<extra>%{fullData.name}</extra>",
        })

    spec = {
        "data": traces,
        "layout": {
            "polar": {
                "radialaxis": {"visible": True, "range": [0, 1], "tickformat": ".0%"},
            },
            "title": {"text": title, "font": {"size": 13}},
            "font": {"family": "Inter, sans-serif", "size": 11},
            "showlegend": True, "legend": {"font": {"size": 10}},
            "margin": {"l": 60, "r": 60, "t": 50, "b": 40},
            "height": 420,
            "paper_bgcolor": "#fff", "plot_bgcolor": "#fff",
        },
    }
    return json.dumps(spec)
