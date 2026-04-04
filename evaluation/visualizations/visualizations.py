"""Grouped bar chart visualization for model performance comparison across
datasets, attack techniques, and metrics. Supports 1D and 2D grouping with
color-coded groups and hatched model patterns.
"""
import matplotlib.pyplot as plt

from typing import Union

from pandas import (
    DataFrame,
    read_csv,
)
from matplotlib.patches import Patch

from config import Config


def setup_plot():
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    plt.subplots_adjust(right=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def setup_axes(ax, tick_positions, tick_labels, ylabel, title, ylim=1.05):
    ax.set_xticks(tick_positions)
    rotation = 0 if len(tick_labels) < 5 else 7.5
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=rotation)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)


def add_legends(ax, group_handles, group_title, model_handles, model_title):
    leg1 = ax.legend(
        handles=group_handles,
        title=group_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        title_fontsize=8,
    )
    ax.legend(
        handles=model_handles,
        title=model_title,
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(leg1)


def plot_performance(
        data: DataFrame,
        metric: Union[str, list[str]],
        group_cols: list[str],
        models: list[str],
        group_colors: Union[list[str], dict[str, str]],
        model_hatches: dict[str, str],
        bar_width: float = 0.8,
        group_gap: float = 1.0,
        inner_gap: float = 0.2,
        alpha_value: float = 0.7,
        edgecolor: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.7),
        ylabel: str = "Score",
        title: str = None,
):
    fig, ax = setup_plot()
    x_pos = 0

    xticks = []
    if isinstance(metric, list) and len(metric) > 1:
        metric_titles = []
        for m in metric:
            group_start = x_pos
            groups = list(data[group_cols[0]].unique())
            for i, group in enumerate(groups):
                for model in models:
                    row = data[(data[group_cols[0]] == group) & (data["Model"] == model)]
                    if row.empty:
                        continue
                    score = row.iloc[0][m]
                    if isinstance(group_colors, list):
                        color = group_colors[i % len(group_colors)]
                    else:
                        color = group_colors.get(group, "#333333")
                    ax.bar(
                        x=x_pos,
                        height=score,
                        width=bar_width,
                        color=color,
                        alpha=alpha_value,
                        edgecolor=edgecolor,
                        hatch=model_hatches.get(model, ""),
                        zorder=3,
                    )
                    x_pos += bar_width
                x_pos += inner_gap
            group_end = x_pos
            xticks.append((group_start + group_end - inner_gap) / 2)
            metric_titles.append(m)
            x_pos += group_gap
        tick_labels = metric_titles
        if title is None:
            title = "Model Performance by Metric, Dataset, and Model"
    else:
        m = metric if isinstance(metric, str) else metric[0]
        if len(group_cols) == 1:
            groups = list(data[group_cols[0]].unique())
            for i, group in enumerate(groups):
                group_start = x_pos
                for model in models:
                    row = data[(data[group_cols[0]] == group) & (data["Model"] == model)]
                    if row.empty:
                        continue
                    score = row.iloc[0][m]
                    if isinstance(group_colors, list):
                        color = group_colors[i % len(group_colors)]
                    else:
                        color = group_colors.get(group, "#333333")
                    ax.bar(
                        x=x_pos,
                        height=score,
                        width=bar_width,
                        color=color,
                        alpha=alpha_value,
                        edgecolor=edgecolor,
                        hatch=model_hatches.get(model, ""),
                        zorder=3,
                    )
                    x_pos += bar_width
                xticks.append((group_start + (x_pos - inner_gap)) / 2)
                x_pos += group_gap
            tick_labels = groups
            if title is None:
                title = f"Model Performance by {group_cols[0]} (Metric: {m})"
        elif len(group_cols) == 2:
            major_gap = 0.5
            majors = list(data[group_cols[0]].unique())
            for major in majors:
                group_start = x_pos
                data_major = data[data[group_cols[0]] == major]
                minors = list(data_major[group_cols[1]].unique())
                for minor in minors:
                    for model in models:
                        row = data_major[
                            (data_major[group_cols[1]] == minor) & (data_major["Model"] == model)]
                        if row.empty:
                            continue
                        score = row.iloc[0][m]
                        color = group_colors.get(minor, "#333333")
                        ax.bar(
                            x=x_pos,
                            height=score,
                            width=bar_width,
                            color=color,
                            alpha=alpha_value,
                            edgecolor=edgecolor,
                            hatch=model_hatches.get(model, ""),
                            zorder=3,
                        )
                        x_pos += bar_width
                    x_pos += inner_gap
                xticks.append((group_start + (x_pos - major_gap)) / 2)
                x_pos += major_gap
            tick_labels = majors
            if title is None:
                title = (
                    f"Model Performance by {group_cols[0]} (x-axis) "
                    f"and {group_cols[1]} (colors); Metric: {m}"
                )
        else:
            raise ValueError("group_cols must be a list of length 1 or 2.")

    setup_axes(ax, xticks, tick_labels, ylabel, title)

    if len(group_cols) == 2:
        minor_unique = list(data[group_cols[1]].unique())
        group_handles = [
            Patch(facecolor=(group_colors[minor_unique.index(g)] if isinstance(group_colors, list)
                             else group_colors.get(g, "#333333")),
                  edgecolor=edgecolor, label=g, alpha=alpha_value)
            for g in minor_unique
        ]
        group_legend_title = group_cols[1]
    else:
        groups = list(data[group_cols[0]].unique())
        group_handles = [
            Patch(
                facecolor=(
                    group_colors[i % len(group_colors)] if isinstance(group_colors, list)
                    else group_colors.get(g, "#333333")
                ),
                edgecolor=edgecolor,
                label=g,
                alpha=alpha_value,
            )
            for i, g in enumerate(groups)
        ]
        group_legend_title = group_cols[0]

    model_handles = [
        Patch(
            facecolor="white",
            edgecolor=edgecolor,
            hatch=model_hatches.get(model, ""),
            label=model,
        )
        for model in models
    ]
    add_legends(ax, group_handles, group_legend_title, model_handles, "Models")
    return fig


def main():
    PREFIX = "../datasets/evaluations"
    datasets = [
        {
            "file": f"{PREFIX}/1_dataset_level.csv",
            "metric": None,
            "group_cols": ["Dataset"],
            "group_colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        },
        {
            "file": f"{PREFIX}/2_fuzzy.csv",
            "metric": "F1",
            "group_cols": ["pii_techniques_str"],
            "group_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        },
        {
            "file": f"{PREFIX}/3_adv.csv",
            "metric": "F1",
            "group_cols": ["content_techniques_str"],
            "group_colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        },
        {
            "file": f"{PREFIX}/4_both.csv",
            "metric": "F1",
            "group_cols": ["pii_techniques_str", "content_techniques_str"],
            "group_colors": {"affix": "#1f77b4", "emojify": "#ff7f0e"},
        },
    ]

    common_params = {
        "models": Config.MODELS,
        "model_hatches": dict(zip(Config.MODELS, ["xxx", "//"])),
        "bar_width": 0.8,
        "group_gap": 1.0,
        "inner_gap": 0.2,
    }

    for dataset in datasets:
        df = read_csv(dataset["file"])
        metric = dataset["metric"] if dataset["metric"] else list(df.columns[2:])
        plot_performance(
            data=df,
            metric=metric,
            group_cols=dataset["group_cols"],
            group_colors=dataset["group_colors"],
            **common_params,
        )
        plt.show()


if __name__ == "__main__":
    main()
