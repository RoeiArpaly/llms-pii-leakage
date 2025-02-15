import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Patch

from config import Config


DATA_PATH = "../datasets/score_results_1.csv"
df = pd.read_csv(DATA_PATH)


def plot_model_performance(
    data,
    metrics,
    datasets,
    models,
    dataset_colors,
    model_hatches,
    bar_width,
    dataset_gap,
    group_gap,
    alpha_value,
    edgecolor,
):

    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    plt.subplots_adjust(right=0.8)
    ax.set_axisbelow(True)  # Ensure grid is drawn behind the bars

    metric_centers = []
    x_pos = 0  # Current x position
    for metric in metrics:
        group_start = x_pos  # Starting x position for this metric group
        for i, dataset in enumerate(datasets):
            for model in models:
                # Extract the score from the data
                score_series = data.loc[
                    (data["Dataset"] == dataset) & (data["Model"] == model), metric
                ]
                if score_series.empty:
                    continue
                score = score_series.iloc[0]
                ax.bar(
                    x=x_pos,
                    height=score,
                    width=bar_width,
                    color=dataset_colors[i],
                    alpha=alpha_value,
                    edgecolor=edgecolor,
                    hatch=model_hatches.get(model, ""),
                    zorder=3,  # Draw bars above grid lines
                )
                x_pos += bar_width
            # Add a gap between dataset groups, except after the last one.
            if i < len(datasets) - 1:
                x_pos += dataset_gap
        group_end = x_pos  # End position for the current metric group
        metric_centers.append((group_start + group_end - dataset_gap) / 2)
        x_pos += group_gap

    ax.set_xticks(metric_centers)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        label="Model Performance by Metric, Dataset, and Model",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    dataset_handles = [
        Patch(
            facecolor=dataset_colors[i],
            edgecolor=edgecolor,
            label=datasets[i],
            alpha=alpha_value,
        )
        for i in range(len(datasets))
    ]
    model_handles = [
        Patch(facecolor="white", edgecolor=edgecolor, hatch=model_hatches[model], label=model)
        for model in models
    ]
    legend1 = ax.legend(
        handles=dataset_handles,
        title="Datasets",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        title_fontsize=8,
    )
    ax.legend(
        handles=model_handles,
        title="Models",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(legend1)
    plt.show()


plot_model_performance(
    data=df,
    metrics=list(df.columns[2:]),
    datasets=list(df["Dataset"].unique()),
    models=Config.MODELS,
    dataset_colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    model_hatches=dict(zip(Config.MODELS, ["///", "xxx", ""])),
    bar_width=0.8,
    dataset_gap=0.2,
    group_gap=1.0,
    alpha_value=0.7,
    edgecolor=(0, 0, 0, 0.7),
)
