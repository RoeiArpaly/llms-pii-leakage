"""Standalone sensitivity analysis script for perplexity threshold tuning.

Loads prediction CSVs from the legacy per-file layout, filters by configured
attacks, and renders the threshold sweep chart.
"""
import json
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pandas import (
    DataFrame,
    read_csv,
)
from sklearn.metrics import confusion_matrix

from config import Config
from utils import infer_json


def _sweep_metrics(y_true, perplexity, thresholds, include_prec=False):
    """Compute recall (and optionally precision) across thresholds."""
    tprs = []
    precisions = []
    for thr in thresholds:
        y_pred = (perplexity > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            tn = fp = fn = tp = 0
            if len(y_true) > 0:
                if all(y_true == 0):
                    tn = len(y_true)
                else:
                    tp = len(y_true)
        else:
            tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tprs.append(tpr)

        if include_prec:
            prec = (
                tp / (tp + fp) if (tp + fp) > 0 else 0.0
            )
            precisions.append(prec)

    return np.array(tprs), np.array(precisions)


def plot_threshold_sweep(
        df: DataFrame,
        thresholds,
        chosen_threshold: float,
        perplexity_col: str = "perplexity",
):
    fig = plt.figure(figsize=(9, 5.8), dpi=100)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "Nimbus Roman",
        ],
        "mathtext.fontset": "cm",
        "font.size": 26,
    })

    colors = {"baseline": "#1f77b4", "fuzzy_adv": "#ff7f0e"}
    handles_labels = {}

    for dataset, group_dataset in df.groupby("dataset"):
        color = colors[dataset]
        ds_name = (
            "Baseline" if dataset == "baseline"
            else "Adversarial"
        )

        for _, group in group_dataset.groupby("model"):
            is_base = dataset == "baseline"
            tprs, precs = _sweep_metrics(
                group["y_true"].values,
                group[perplexity_col].values,
                thresholds,
                include_prec=is_base,
            )

            label_recall = f"{ds_name} Recall"
            if label_recall not in handles_labels:
                h, = plt.plot(
                    thresholds, tprs * 100,
                    color=color, lw=2, linestyle="-",
                    label=label_recall,
                )
                handles_labels[label_recall] = h
            else:
                plt.plot(
                    thresholds, tprs * 100,
                    color=color, lw=2, linestyle="-",
                )

            if is_base:
                label_prec = f"{ds_name} Precision"
                if label_prec not in handles_labels:
                    h, = plt.plot(
                        thresholds, precs * 100,
                        color=color, lw=2, linestyle="--",
                        label=label_prec, alpha=0.75,
                    )
                    handles_labels[label_prec] = h
                else:
                    plt.plot(
                        thresholds, precs * 100,
                        color=color, lw=2, linestyle="--",
                        alpha=0.75,
                    )

    h_thresh = plt.axvline(
        x=chosen_threshold, color="k",
        linestyle="--", lw=1.5, alpha=0.75,
    )
    handles_labels["Chosen Threshold"] = h_thresh

    plt.xlabel("Perplexity Threshold", labelpad=15)
    plt.ylabel("Metric Value [%]")
    plt.ylim(-5, 105)
    plt.title(
        "Sensitivity Analysis of Perplexity Threshold",
        fontsize=26, pad=20,
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=16)

    ordered = [
        "Baseline Precision", "Baseline Recall",
        "Adversarial Recall", "Chosen Threshold",
    ]
    handles = [
        handles_labels[k] for k in ordered
        if k in handles_labels
    ]
    plt.legend(
        handles, ordered, fontsize=14,
        bbox_to_anchor=(0.965, 0.9),
    )

    def latex_formatter(x, pos):
        if math.isclose(x, 1.0):
            return "1"
        offset = x - 1
        base, exponent = f"{offset:.0e}".split('e')
        exponent = int(exponent)
        return (
            r"${} {{1 + {} \cdot 10^{{{}}}}}$"
            .format(
                "" if offset > 0 else "",
                base, exponent,
            )
        )

    plt.gca().yaxis.set_major_locator(
        mticker.MultipleLocator(20),
    )
    plt.gca().xaxis.set_major_formatter(
        mticker.FuncFormatter(latex_formatter),
    )
    plt.tight_layout()
    return fig


def _format_threshold(x):
    """Format threshold for Plotly tick labels."""
    if math.isclose(x, 1.0):
        return "1"
    offset = x - 1
    base, exponent = f"{offset:.0e}".split("e")
    return f"1+{base}e{int(exponent)}"


def plot_threshold_sweep_plotly(
        df: DataFrame,
        thresholds,
        chosen_threshold: float,
        perplexity_col: str = "perplexity",
) -> str:
    """Return a Plotly JSON spec for the threshold sweep."""
    colors = {"baseline": "#1f77b4", "fuzzy_adv": "#ff7f0e"}
    traces = []
    seen = set()

    for dataset, group_dataset in df.groupby("dataset"):
        color = colors.get(dataset, "#333")
        ds_name = (
            "Baseline" if dataset == "baseline"
            else "Adversarial"
        )

        for _, group in group_dataset.groupby("model"):
            is_base = dataset == "baseline"
            tprs, precs = _sweep_metrics(
                group["y_true"].values,
                group[perplexity_col].values,
                thresholds,
                include_prec=is_base,
            )

            thr_list = thresholds.tolist()

            label_recall = f"{ds_name} Recall"
            show = label_recall not in seen
            seen.add(label_recall)
            traces.append({
                "x": thr_list,
                "y": (tprs * 100).tolist(),
                "mode": "lines",
                "name": label_recall,
                "line": {"color": color, "width": 2},
                "showlegend": show,
            })

            if is_base:
                label_prec = f"{ds_name} Precision"
                show = label_prec not in seen
                seen.add(label_prec)
                traces.append({
                    "x": thr_list,
                    "y": (precs * 100).tolist(),
                    "mode": "lines",
                    "name": label_prec,
                    "line": {
                        "color": color, "width": 2,
                        "dash": "dash",
                    },
                    "opacity": 0.75,
                    "showlegend": show,
                })

    # Chosen threshold vertical line
    traces.append({
        "x": [chosen_threshold, chosen_threshold],
        "y": [0, 100],
        "mode": "lines",
        "name": "Chosen Threshold",
        "line": {
            "color": "black", "width": 1.5,
            "dash": "dash",
        },
        "opacity": 0.75,
    })

    # Tick values — ~6 evenly spaced across range
    thr_min = float(thresholds[0])
    thr_max = float(thresholds[-1])
    n_ticks = 6
    tick_vals = np.linspace(thr_min, thr_max, n_ticks).tolist()
    tick_text = [_format_threshold(v) for v in tick_vals]

    layout = {
        "title": {
            "text": "Sensitivity Analysis of Perplexity Threshold",
        },
        "xaxis": {
            "title": "Perplexity Threshold",
            "tickvals": tick_vals,
            "ticktext": tick_text,
        },
        "yaxis": {
            "title": "Metric Value [%]",
            "range": [-5, 105],
            "dtick": 20,
        },
        "legend": {
            "x": 0.98, "y": 0.95,
            "xanchor": "right",
        },
        "margin": {"l": 60, "r": 20, "t": 50, "b": 60},
    }

    return json.dumps({"data": traces, "layout": layout})


def main():
    from pathlib import Path

    predictions_path = Path("datasets/predictions.csv")
    dataset_path = Path("datasets/dataset.csv")

    if not predictions_path.exists() or not dataset_path.exists():
        print("Missing datasets/predictions.csv or datasets/dataset.csv")
        return

    predictions = read_csv(predictions_path).apply(infer_json)
    dataset = read_csv(dataset_path).apply(infer_json)

    model = "gpt-4o-mini-defend"
    preds = predictions[predictions["model"] == model]
    if preds.empty:
        print(f"No predictions found for {model}")
        return

    merged = preds.merge(
        dataset[["uid", "pii_spans", "category", "attack_target"]],
        on="uid", how="left",
    )

    pii_ok = set(Config.ATTACKS)
    ctx_ok = set(Config.CONTENT_ATTACKS)
    merged = merged[merged["attack_target"].apply(
        lambda x: (
            all(v in pii_ok for v in x.get("pii", []))
            and all(v in ctx_ok for v in x.get("context", []))
        ) if isinstance(x, dict) else True
    )]

    merged["y_true"] = merged["pii_spans"].apply(lambda x: len(x) > 0)
    merged["dataset"] = merged["attack_target"].apply(
        lambda x: "fuzzy_adv" if isinstance(x, dict) else "baseline"
    )

    plot_threshold_sweep(
        df=merged,
        thresholds=np.arange(1, 1.000005, 0.00000001),
        chosen_threshold=Config.PERPLEXITY_THRESHOLD,
    )
    plt.show()


if __name__ == "__main__":
    main()
