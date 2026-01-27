import numpy as np
import matplotlib.pyplot as plt

from pandas import (
    concat,
    read_csv,
    set_option,
)
from sklearn.metrics import confusion_matrix

from config import Config
from utils import infer_json


set_option("display.max_columns", None)


MODELS = ["gpt-4o-mini-defend"]

datasets = []
for dataset in ["baseline", "fuzzy_adv"]:
    for model in MODELS:
        data = read_csv(f"datasets/predictions/{dataset}_{model}.csv").apply(infer_json)
        if "fuzzy_techniques" in data.columns:
            data = data[data["fuzzy_techniques"].apply(
                lambda x: all([v in Config.ATTACKS for v in x]))
            ]
        if "adv_content_techniques" in data.columns:
            data = data[data["adv_content_techniques"].apply(
                lambda x: all([v in Config.CONTENT_ATTACKS for v in x]))
            ]

        data["y_true"] = data["pii_spans"].apply(lambda x: len(x) > 0)
        data["y_pred"] = data["prediction"].apply(lambda x: len(x) > 0)
        data["match"] = data["y_true"] == data["y_pred"]

        data["dataset"] = dataset
        data["model"] = model
        datasets.append(data)


raw = concat(datasets, ignore_index=True)


def plot_threshold_sweep_usenix(
        df,
        thresholds,
        chosen_threshold: float,
        perplexity_col: str = "perplexity",
):
    """
    Sensitivity analysis plot for USENIX-style paper.
    """

    plt.figure(figsize=(9, 5.8), dpi=100)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "mathtext.fontset": "cm",  # Computer Modern for LaTeX-like math
        "font.size": 26,
    })

    colors = {"baseline": "#1f77b4", "fuzzy_adv": "#ff7f0e"}
    linestyle_map = {"Precision": "--", "F1": "-."}  # Recall forced solid

    # Track handles and labels
    handles_labels = {}

    for dataset, group_dataset in df.groupby("dataset"):
        color = colors[dataset]
        dataset_name = "Baseline" if dataset == "baseline" else "Adversarial"

        for model, group in group_dataset.groupby("model"):
            y_true = group["y_true"].values
            perplexity = group[perplexity_col].values

            tprs = []
            precisions = []
            f1s = []

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

                if dataset == "baseline":
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    precisions.append(precision)
                    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
                    f1s.append(f1)

            # Plot Recall
            label_recall = f"{dataset_name} Recall"
            if label_recall not in handles_labels:
                h_recall, = plt.plot(
                    thresholds,
                    np.array(tprs) * 100,
                    color=color,
                    lw=2,
                    linestyle="-",
                    label=label_recall,
                )
                handles_labels[label_recall] = h_recall
            else:
                plt.plot(thresholds, np.array(tprs) * 100, color=color, lw=2, linestyle="-")

            if dataset == "baseline":
                label_prec = f"{dataset_name} Precision"
                if label_prec not in handles_labels:
                    h_prec, = plt.plot(
                        thresholds,
                        np.array(precisions) * 100,
                        color=color,
                        lw=2,
                        linestyle=linestyle_map["Precision"],
                        label=label_prec,
                        alpha=0.75,
                    )
                    handles_labels[label_prec] = h_prec
                else:
                    plt.plot(
                        thresholds,
                        np.array(precisions) * 100,
                        color=color,
                        lw=2,
                        linestyle=linestyle_map["Precision"],
                        alpha=0.75,
                    )

    # Chosen threshold line
    h_thresh = plt.axvline(x=chosen_threshold, color="k", linestyle="--", lw=1.5, alpha=0.75)
    handles_labels["Chosen Threshold"] = h_thresh

    # Axes & grid
    plt.xlabel("Perplexity Threshold", labelpad=15)
    plt.ylabel("Metric Value [%]")
    plt.ylim(-5, 105)
    plt.title("Sensitivity Analysis of Perplexity Threshold", fontsize=26, pad=20)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Set fontsize of xtick labels
    plt.xticks(fontsize=16)

    # Legend ordered manually
    ordered_labels = [
        "Baseline Precision", "Baseline Recall", "Adversarial Recall", "Chosen Threshold",
    ]
    ordered_handles = [handles_labels[lbl] for lbl in ordered_labels if lbl in handles_labels]
    plt.legend(ordered_handles, ordered_labels, fontsize=14, bbox_to_anchor=(0.965, 0.9))

    import matplotlib.ticker as mticker
    import math

    def latex_formatter(x, pos):
        if math.isclose(x, 1.0):
            return "1"
        # Calculate offset
        offset = x - 1
        # Get scientific notation parts: 2e-06 -> 2 and -6
        # Note: We use .1e to enforce scientific notation consistently
        base, exponent = f"{offset:.0e}".split('e')
        # Clean up the exponent (remove leading zeros, e.g., -06 -> -6)
        exponent = int(exponent)
        # Return LaTeX string
        return r"${} {{1 + {} \cdot 10^{{{}}}}}$".format(
            "" if offset > 0 else "",  # visual spacer if needed
            base,
            exponent
        )

    # Make sure there the y ticks are [0, 20, 40, 60, 80, 100]
    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(20))
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(latex_formatter))
    plt.tight_layout()
    plt.show()


plot_threshold_sweep_usenix(
    df=raw,
    thresholds=np.arange(1, 1.000005, 0.00000001),
    chosen_threshold=Config.PERPLEXITY_THRESHOLD,
)
