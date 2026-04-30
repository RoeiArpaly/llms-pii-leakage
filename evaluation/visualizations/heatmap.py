"""Heatmap visualizations (static matplotlib + interactive Plotly JSON) showing
model performance scores across attack categories or dataset splits.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame

from evaluation.report.config import MODEL_ORDER, display_name

_COLORSCALE = [
    [0.0, "#f7f7f7"], [0.2, "#d1e5f0"], [0.4, "#92c5de"],
    [0.6, "#4393c3"], [0.8, "#2166ac"], [1.0, "#053061"],
]


def _pivot(
    df: DataFrame, metric: str, index_col: str, cols_col: str = "Model",
) -> tuple[list[str], list[str], np.ndarray]:
    if cols_col == "Model" and "Model" in df.columns:
        col_vals = [m for m in MODEL_ORDER if m in df["Model"].unique()]
    else:
        col_vals = list(df[cols_col].unique())
    groups = list(df[index_col].unique())
    matrix = np.full((len(groups), len(col_vals)), np.nan)
    for i, group in enumerate(groups):
        for j, col in enumerate(col_vals):
            row = df[(df[index_col] == group) & (df[cols_col] == col)]
            if not row.empty:
                matrix[i, j] = row.iloc[0][metric]
    return groups, col_vals, matrix


def heatmap(df: DataFrame, metric: str, index_col: str,
            cols_col: str = "Model") -> plt.Figure:
    groups, models, matrix = _pivot(df, metric, index_col, cols_col)
    display_models = [display_name(m) for m in models]
    display_groups = [display_name(g) for g in groups]

    fig, ax = plt.subplots(
        figsize=(max(9, len(models) * 1.35), max(3.5, len(groups) * 0.75)), dpi=200,
    )
    cmap = LinearSegmentedColormap.from_list(
        "academic", [c[1] for c in _COLORSCALE]
    )
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(display_models)))
    ax.set_xticklabels(display_models, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(display_groups)))
    ax.set_yticklabels(display_groups, fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(groups)):
        for j in range(len(models)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "#ffffff" if val > 0.45 else "#1a1a1a"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=20)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label(display_name(metric), fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def heatmap_plotly(df: DataFrame, metric: str, index_col: str,
                   cols_col: str = "Model") -> str:
    groups, models, matrix = _pivot(df, metric, index_col, cols_col)
    d_models = [display_name(m) for m in models]
    d_groups = [display_name(g) for g in groups]
    z = [[None if np.isnan(v) else round(v, 4) for v in row] for row in matrix]
    text = [[f"{v:.0%}" if v is not None else "" for v in row] for row in z]
    spec = {
        "data": [{
            "z": z, "x": d_models, "y": d_groups, "type": "heatmap",
            "colorscale": _COLORSCALE, "zmin": 0, "zmax": 1,
            "text": text, "texttemplate": "%{text}",
            "textfont": {"size": 12},
            "hovertemplate": "%{y}<br>%{x}<br>%{z:.1%}<extra></extra>",
            "showscale": True,
            "colorbar": {"title": display_name(metric), "tickformat": ".0%"},
        }],
        "layout": {
            "margin": {
                "l": max(140, max((len(g) for g in d_groups), default=10) * 7),
                "r": 40, "t": 30, "b": 100,
            },
            "xaxis": {"side": "bottom", "tickangle": -40},
            "yaxis": {"autorange": "reversed"},
            "font": {"family": "Inter, sans-serif", "size": 11},
            "plot_bgcolor": "#fff", "paper_bgcolor": "#fff",
            "height": max(280, len(groups) * 32 + 160),
        },
    }
    return json.dumps(spec)
