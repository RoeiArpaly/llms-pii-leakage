"""Grouped bar chart visualization for model performance comparison.

Provides Plotly interactive grouped bar charts showing per-model metrics
(F1, Recall, Precision) grouped by attack category.
"""
import json

from pandas import DataFrame

from evaluation.report.config import (
    MODEL_COLORS,
    display_name,
)
from evaluation.visualizations.utils import extract_series


def grouped_bar_plotly(
    df: DataFrame,
    metric: str,
    group_col: str,
) -> str:
    """Build a Plotly grouped bar chart JSON spec.

    X-axis: attack categories, bars grouped by model, colored by model.
    """
    models, categories, series = extract_series(df, metric, group_col)
    d_cats = [display_name(c) for c in categories]

    traces = []
    for model in models:
        color = MODEL_COLORS.get(model, "#888888")
        traces.append({
            "type": "bar",
            "x": d_cats,
            "y": series[model],
            "name": display_name(model),
            "marker": {"color": color},
            "hovertemplate": (
                "%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>"
            ),
        })

    spec = {
        "data": traces,
        "layout": {
            "barmode": "group",
            "yaxis": {
                "title": display_name(metric),
                "range": [0, 1.08],
                "tickformat": ".0%",
            },
            "xaxis": {"tickangle": -25},
            "font": {"family": "Inter, sans-serif", "size": 12},
            "showlegend": True,
            "legend": {"font": {"size": 10}, "x": 1.02, "y": 1},
            "margin": {"l": 60, "r": 180, "t": 30, "b": 100},
            "height": 450,
            "paper_bgcolor": "#fff",
            "plot_bgcolor": "#fff",
            "hovermode": "x unified",
        },
    }
    return json.dumps(spec)
