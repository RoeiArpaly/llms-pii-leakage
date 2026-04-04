"""Matplotlib style configuration and figure-to-base64 conversion for
embedding static charts in the HTML report.
"""
import base64

from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#999999", "axes.linewidth": 0.6,
        "axes.labelcolor": "#333333", "axes.labelsize": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.grid": True, "grid.color": "#e8e8e8",
        "grid.linewidth": 0.5, "grid.linestyle": "--",
        "xtick.color": "#555555", "ytick.color": "#555555",
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9, "legend.frameon": True,
        "legend.edgecolor": "#cccccc", "legend.fancybox": False, "legend.framealpha": 0.95,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
