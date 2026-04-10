"""HTML report generator: aggregates evaluation scores, builds interactive
visualizations (heatmaps, radar charts, line charts), and renders a
self-contained HTML dashboard with dark/light theme and Plotly interactivity.
"""
import json
import webbrowser

from pathlib import Path

import numpy as np

from pandas import (
    DataFrame,
    read_csv,
)

from config import Config
from evaluation.report.config import (
    display_name,
    sort_models,
)
from evaluation.report.html import (
    CSS,
    DATA_SECTIONS,
    JS,
    PLOTLY_CDN,
    _chart_panel,
    render_performance_page,
    render_static_section,
    styled_table,
)
from evaluation.scoring import SPANS_METRICS
from evaluation.visualizations.sensitivity_analysis_perplexity import (
    plot_threshold_sweep,
    plot_threshold_sweep_plotly,
)
from evaluation.visualizations.style import apply_style, fig_to_base64
from utils import infer_json


DATASET_PATH = Path("datasets/dataset.csv")
PREDICTIONS_PATH = Path("datasets/predictions.csv")


def _load_report_dataset() -> DataFrame:
    """Load dataset filtered to UIDs present in predictions.

    When the pipeline ran with --sample, predictions cover a subset of the
    dataset. Filtering ensures overview stats match the evaluated data.
    """
    if not DATASET_PATH.exists():
        return DataFrame()
    dataset = read_csv(DATASET_PATH).apply(infer_json)
    if PREDICTIONS_PATH.exists():
        pred_uids = set(read_csv(PREDICTIONS_PATH, usecols=["uid"])["uid"])
        if len(pred_uids) < len(dataset):
            dataset = dataset[dataset["uid"].isin(pred_uids)]
    return dataset


def _compute_aggregated_scores(data: DataFrame, groupby_cols: list[str] = None) -> DataFrame:
    if groupby_cols:
        data = data.groupby(groupby_cols)[SPANS_METRICS].sum().reset_index()
    else:
        data = DataFrame([data[SPANS_METRICS].sum()])
    data["Precision"] = data["true_positive"] / (data["true_positive"] + data["false_positive"])
    data["Recall"] = data["true_positive"] / (data["true_positive"] + data["false_negative"])
    data["F1"] = 2 * data["Precision"] * data["Recall"] / (data["Precision"] + data["Recall"])
    return data.drop(columns=SPANS_METRICS).fillna(0)


def _compute_report_data() -> dict[str, DataFrame]:
    """Compute performance data on-the-fly from predictions
    + dataset (no evaluations.csv needed)."""
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return {}

    from evaluation import spans_scorer

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)

    data = predictions.merge(
        dataset[["uid", "category", "pii_spans", "attack_target"]],
        on="uid", how="left",
    )

    # Compute span scores on-the-fly
    data["spans_score"] = data.apply(
        lambda row: spans_scorer(
            spans_true=row["pii_spans"],
            spans_pred=row["prediction"],
            match_level=Config.MATCH_LEVEL,
            method=Config.METHOD,
        ),
        axis=1,
    )

    for col in SPANS_METRICS:
        data[col] = data["spans_score"].apply(
            lambda x: x.get(col) if isinstance(x, dict) else 0,
        )

    data["pii_techniques_str"] = data["attack_target"].apply(
        lambda x: (
            "_".join(x["pii"])
            if isinstance(x, dict) and x.get("pii")
            else None
        ),
    )
    data["content_techniques_str"] = data["attack_target"].apply(
        lambda x: (
            "_".join(x["context"])
            if isinstance(x, dict) and x.get("context")
            else None
        ),
    )

    results = {}

    def _agg_by(source, group_col, result_key, label_col):
        filtered = source[source[group_col].notna()]
        if filtered.empty:
            return
        rows = []
        for group_val in filtered[group_col].unique():
            group_data = filtered[
                filtered[group_col] == group_val
            ]
            for model in group_data["model"].unique():
                agg = _compute_aggregated_scores(
                    group_data[group_data["model"] == model],
                )
                agg[label_col] = group_val
                agg["Model"] = model
                rows.append(agg.iloc[0].to_dict())
        if rows:
            results[result_key] = DataFrame.from_records(
                rows,
            )

    _agg_by(
        data, "pii_techniques_str",
        "fuzzy", "fuzzy_techniques",
    )
    _agg_by(
        data, "content_techniques_str",
        "adv", "adv_content_techniques",
    )

    return results


def _build_overview(dataset: DataFrame) -> str:
    counts = dataset["category"].value_counts().to_dict()
    n_negative = counts.get("negative", 0)
    n_hard_neg = counts.get("hard_negative", 0)
    n_total = len(dataset)

    positive = dataset[dataset["category"] == "positive"]

    def _has_pii(t):
        return isinstance(t, dict) and bool(t.get("pii"))

    def _has_ctx(t):
        return isinstance(t, dict) and bool(t.get("context"))

    n_clean = positive[~positive["attack_target"].apply(
        lambda x: isinstance(x, dict),
    )].shape[0]
    n_direct = positive[positive["attack_target"].apply(
        lambda t: _has_pii(t) and not _has_ctx(t),
    )].shape[0]
    n_both = positive[positive["attack_target"].apply(
        lambda t: _has_ctx(t),
    )].shape[0]

    # PII type breakdown
    pii_type_counts = {}
    for spans in positive["pii_spans"]:
        if not isinstance(spans, list):
            continue
        for span in spans:
            if isinstance(span, dict):
                t = display_name(
                    span.get("type", "unknown"),
                )
                pii_type_counts[t] = (
                    pii_type_counts.get(t, 0) + 1
                )

    # ── Stacked bar helper ──
    def _stacked_bar(segments, total, height="28px"):
        """Render a stacked horizontal bar.

        segments: list of (label, count, color)
        """
        if total == 0:
            return ""
        segs_html = ""
        for label, count, color in segments:
            pct = count / total * 100
            if pct < 0.5:
                continue
            segs_html += (
                f'<div title="{label}: {count} '
                f'({pct:.1f}%)" style="width:{pct}%;'
                f'background:{color};height:100%">'
                f'</div>'
            )
        legend = " ".join(
            f'<span style="display:inline-flex;'
            f'align-items:center;gap:3px">'
            f'<span style="width:10px;height:10px;'
            f'border-radius:2px;background:{c};'
            f'display:inline-block"></span>'
            f'<span>{lbl}</span>'
            f'<strong>{cnt}</strong></span>'
            for lbl, cnt, c in segments if cnt > 0
        )
        return (
            f'<div style="display:flex;border-radius:4px;'
            f'overflow:hidden;height:{height};'
            f'margin:0.3rem 0">{segs_html}</div>'
            f'<div style="font-size:0.75rem;'
            f'color:var(--text-muted);margin-top:0.2rem;'
            f'display:flex;flex-wrap:wrap;gap:0.6rem">'
            f'{legend}</div>'
        )

    # Dataset composition bar
    dataset_bar = _stacked_bar([
        ("Negatives", n_negative, "rgba(69,117,180,0.45)"),
        ("Hard Negatives", n_hard_neg, "rgba(116,173,209,0.45)"),
        ("Positives", n_clean, "rgba(253,174,97,0.45)"),
        ("Direct Attack Positives", n_direct, "rgba(244,109,67,0.45)"),
        ("Direct + Indirect Attack Positives", n_both, "rgba(215,48,39,0.45)"),
    ], n_total)

    # PII types bar (narrower)
    pii_bar = ""
    if pii_type_counts:
        pii_colors = [
            "rgba(102,178,102,0.55)",
            "rgba(178,102,178,0.55)",
            "rgba(102,178,178,0.55)",
            "rgba(178,153,102,0.55)",
            "rgba(153,102,153,0.55)",
        ]
        pii_total = sum(pii_type_counts.values())
        pii_segments = [
            (k, v, pii_colors[i % len(pii_colors)])
            for i, (k, v) in enumerate(sorted(
                pii_type_counts.items(),
                key=lambda x: -x[1],
            ))
        ]
        pii_bar = (
            f'<div style="margin-top:0.8rem">'
            f'<span style="font-size:0.82rem;'
            f'color:var(--text-muted)">'
            f'PII Types ({pii_total} spans)</span>'
            f'{_stacked_bar(pii_segments, pii_total, "18px")}'
            f'</div>'
        )

    header = (
        f'<div style="font-size:1.6rem;font-weight:700;'
        f'color:var(--accent);'
        f'font-family:var(--font-heading);'
        f'margin-bottom:0.3rem">'
        f'{n_total} <span style="font-size:0.85rem;'
        f'color:var(--text-muted);font-weight:400">'
        f'Total Samples</span></div>'
    )

    data_card = (
        f'<h3 style="margin:0 0 0.5rem;font-size:0.95rem">'
        f'Data Card</h3>'
        f'{header}{dataset_bar}{pii_bar}'
    )
    return data_card


def _build_leaderboard() -> str | None:
    """Build two leaderboard tables (Base / Shield) with F1."""
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)

    merged = predictions.merge(
        dataset[["uid", "category", "pii_spans", "attack_target"]],
        on="uid", how="left",
    )

    def _segment(row):
        cat = row["category"]
        if cat == "negative":
            return "Negative"
        if cat == "hard_negative":
            return "Hard Negative"
        t = row["attack_target"]
        if not isinstance(t, dict):
            return "Clean Positives"
        has_pii = bool(t.get("pii"))
        has_ctx = bool(t.get("context"))
        if has_pii and not has_ctx:
            return "Direct Attack"
        if has_ctx:
            return "Direct + Indirect"
        return "Clean Positives"

    merged["segment"] = merged.apply(_segment, axis=1)

    rows = []
    for model in merged["model"].unique():
        m = merged[merged["model"] == model]
        row = {"Model": model}

        # Per-segment recall
        total_tp = 0
        total_pos = 0
        for seg in [
            "Clean Positives", "Direct Attack",
            "Direct + Indirect",
        ]:
            seg_data = m[m["segment"] == seg]
            if seg_data.empty:
                row[f"Recall\n{seg}"] = None
                continue
            tp = seg_data.apply(
                lambda r: (
                    len(r["pii_spans"]) > 0
                    and len(r["prediction"]) > 0
                ) if (
                    isinstance(r["pii_spans"], list)
                    and isinstance(r["prediction"], list)
                ) else False, axis=1,
            ).sum()
            total = seg_data["pii_spans"].apply(
                lambda s: (
                    len(s) > 0
                    if isinstance(s, list) else False
                ),
            ).sum()
            row[f"Recall\n{seg}"] = (
                tp / total if total > 0 else 0.0
            )
            total_tp += tp
            total_pos += total

        # Per-segment precision (TNR)
        total_tn = 0
        total_neg = 0
        for seg in ["Negative", "Hard Negative"]:
            seg_data = m[m["segment"] == seg]
            if seg_data.empty:
                row[f"Precision\n{seg}"] = None
                continue
            n_total = len(seg_data)
            n_tn = seg_data["prediction"].apply(
                lambda p: (
                    len(p) == 0
                    if isinstance(p, list) else True
                ),
            ).sum()
            row[f"Precision\n{seg}"] = (
                n_tn / n_total if n_total > 0 else 0.0
            )
            total_tn += n_tn
            total_neg += n_total

        # Overall F1 (binary detection level)
        recall = (
            total_tp / total_pos
            if total_pos > 0 else 0.0
        )
        fp = total_neg - total_tn
        prec = (
            total_tp / (total_tp + fp)
            if (total_tp + fp) > 0 else 0.0
        )
        f1 = (
            2 * prec * recall / (prec + recall)
            if (prec + recall) > 0 else 0.0
        )
        row["F1"] = f1
        rows.append(row)

    if not rows:
        return None

    df = DataFrame.from_records(rows)

    from evaluation.report.config import model_sort_key

    # Exclude pii-shield — it gets its own card
    df = df[df["Model"] != "pii-shield"]

    base_df = df[~df["Model"].str.endswith("-defend")].copy()
    base_df["_ord"] = base_df["Model"].apply(model_sort_key)
    base_df = base_df.sort_values("_ord").drop(columns=["_ord"])

    shield_df = df[df["Model"].str.endswith("-defend")].copy()
    shield_df["Model"] = shield_df["Model"].str.removesuffix("-defend")
    shield_df["_ord"] = shield_df["Model"].apply(model_sort_key)
    shield_df = shield_df.sort_values("_ord").drop(columns=["_ord"])

    if base_df.empty and shield_df.empty:
        return None

    base_html = styled_table(base_df) if not base_df.empty else ""
    shield_html = styled_table(shield_df) if not shield_df.empty else ""

    toggle = (
        '<div style="display:flex;justify-content:center;'
        'padding:1.2rem 0;margin:0.5rem 0">'
        '<button type="button" id="shield-toggle" '
        'style="background:none;border:none;'
        'cursor:pointer;padding:0;'
        'display:inline-flex;flex-direction:column;'
        'align-items:center;gap:4px;'
        'transition:transform 0.2s" '
        'title="Apply Defense Hardening">'
        '<svg width="84" height="94" '
        'viewBox="0 0 24 28" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<path id="shield-path" '
        'd="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 '
        '5.16-1.26 9-6.45 9-12V5L12 1z" '
        'stroke="var(--border)" stroke-width="1.2" '
        'fill="none" style="transition:0.3s"/>'
        '<text x="12" y="12.5" text-anchor="middle" '
        'font-size="3.2" font-weight="700" '
        'fill="var(--text-muted)" '
        'id="shield-text-top" '
        'style="transition:0.3s;user-select:none">'
        'SENTINEL</text></svg>'
        '<span id="shield-hint" '
        'style="font-size:0.75rem;'
        'color:var(--text-muted);'
        'transition:0.3s">Click to Apply</span>'
        '</button></div>'
    )
    return (
        f'{toggle}'
        f'<div id="leaderboard-base" '
        f'style="display:block">{base_html}</div>'
        f'<div id="leaderboard-shield" '
        f'style="display:none">{shield_html}</div>'
    )


def _build_shield_card() -> str | None:
    """Build a dedicated PII Shield summary card with cascade tier breakdown."""
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)
    shield_rows = predictions[predictions["model"] == "pii-shield"]
    if shield_rows.empty:
        return None

    dataset = _load_report_dataset()
    merged = shield_rows.merge(
        dataset[["uid", "category", "pii_spans"]], on="uid", how="left",
    )

    # Overall metrics
    pos = merged[merged["category"] == "positive"]
    neg = merged[merged["category"].isin(["negative", "hard_negative"])]

    total_pos = pos["pii_spans"].apply(
        lambda s: len(s) > 0 if isinstance(s, list) else False,
    ).sum()
    total_tp = pos.apply(
        lambda r: (
            len(r["pii_spans"]) > 0 and len(r["prediction"]) > 0
        ) if (
            isinstance(r["pii_spans"], list)
            and isinstance(r["prediction"], list)
        ) else False, axis=1,
    ).sum()
    total_neg_count = len(neg)
    total_tn = neg["prediction"].apply(
        lambda p: len(p) == 0 if isinstance(p, list) else True,
    ).sum()

    recall = total_tp / total_pos if total_pos > 0 else 0.0
    fp = total_neg_count - total_tn
    prec = total_tp / (total_tp + fp) if (total_tp + fp) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

    # Cascade tier breakdown — which tier caught detections
    detected = shield_rows[shield_rows["prediction"].apply(
        lambda p: len(p) > 0 if isinstance(p, list) else False,
    )]
    n_detected = len(detected)

    tier_counts = {}
    if "detector" in detected.columns and n_detected > 0:
        tier_counts = detected["detector"].value_counts().to_dict()

    # Tier bar segments
    tier_segments = []
    tier_colors = [
        "rgba(69,117,180,0.6)",
        "rgba(116,173,209,0.6)",
        "rgba(102,178,102,0.6)",
        "rgba(178,153,102,0.6)",
        "rgba(253,174,97,0.6)",
    ]
    from evaluation.shield_eval import SHIELD_CASCADE
    for i, tier in enumerate(SHIELD_CASCADE):
        count = tier_counts.get(tier, 0)
        if count > 0:
            tier_segments.append((
                display_name(tier.removesuffix("-defend")),
                count,
                tier_colors[i % len(tier_colors)],
            ))

    tier_bar_html = ""
    if tier_segments and n_detected > 0:
        segs = ""
        for label, count, color in tier_segments:
            pct = count / n_detected * 100
            segs += (
                f'<div title="{label}: {count} ({pct:.1f}%)" '
                f'style="width:{pct}%;background:{color};'
                f'height:100%;display:flex;align-items:center;'
                f'justify-content:center;font-size:0.7rem;'
                f'color:#333;font-weight:500;overflow:hidden;'
                f'white-space:nowrap">'
                f'{pct:.0f}%</div>'
            )
        legend = " ".join(
            f'<span style="display:inline-flex;align-items:center;gap:3px">'
            f'<span style="width:10px;height:10px;border-radius:2px;'
            f'background:{c};display:inline-block"></span>'
            f'<span>{lbl}</span> <strong>{cnt}</strong></span>'
            for lbl, cnt, c in tier_segments
        )
        tier_bar_html = (
            f'<div style="margin-top:0.8rem">'
            f'<span style="font-size:0.82rem;color:var(--text-muted)">'
            f'Cascade Breakdown ({n_detected} detections)</span>'
            f'<div style="display:flex;border-radius:4px;overflow:hidden;'
            f'height:24px;margin:0.3rem 0">{segs}</div>'
            f'<div style="font-size:0.75rem;color:var(--text-muted);'
            f'margin-top:0.2rem;display:flex;flex-wrap:wrap;gap:0.6rem">'
            f'{legend}</div>'
            f'</div>'
        )

    # Stat boxes
    stats = [
        ("F1", f"{f1:.1%}"),
        ("Recall", f"{recall:.1%}"),
        ("Precision", f"{prec:.1%}"),
        ("Tiers", str(len(tier_segments))),
    ]
    stat_boxes = ""
    for label, value in stats:
        stat_boxes += (
            f'<div style="text-align:center;padding:0.6rem 0.4rem">'
            f'<div style="font-size:1.4rem;font-weight:700;'
            f'color:var(--accent);font-family:var(--font-heading);'
            f'line-height:1.2">{value}</div>'
            f'<div style="font-size:0.75rem;color:var(--text-muted);'
            f'margin-top:0.15rem">{label}</div>'
            f'</div>'
        )

    return (
        f'<div style="margin-top:1.5rem;border:1px solid var(--border);'
        f'border-radius:var(--radius);padding:1rem 1.2rem;'
        f'background:var(--card-bg)">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;'
        f'margin-bottom:0.5rem">'
        f'<svg width="20" height="22" viewBox="0 0 24 28" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 '
        f'5.16-1.26 9-6.45 9-12V5L12 1z" '
        f'stroke="var(--accent)" stroke-width="1.5" '
        f'fill="var(--accent)" fill-opacity="0.15"/></svg>'
        f'<h3 style="margin:0;font-size:0.95rem">PII Shield</h3>'
        f'<span style="font-size:0.75rem;color:var(--text-muted)">'
        f'Cascading Defense</span>'
        f'</div>'
        f'<p style="font-size:0.78rem;color:var(--text-muted);'
        f'margin:0 0 0.5rem;line-height:1.4">'
        f'Multi-tier cascade that checks each detector in order and '
        f'returns on the first detection. Uses defend-preprocessed inputs.</p>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
        f'gap:0.5rem">{stat_boxes}</div>'
        f'{tier_bar_html}'
        f'</div>'
    )


def _build_fp_analysis() -> dict[str, DataFrame] | None:
    """Build FP tables keyed by category: 'all', 'negative', 'hard_negative'."""
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)

    # Exclude pii-shield — it has its own summary card
    predictions = predictions[predictions["model"] != "pii-shield"]

    neg_uids = dataset[dataset["category"].isin(["negative", "hard_negative"])]["uid"]
    neg_preds = predictions[predictions["uid"].isin(neg_uids)].merge(
        dataset[["uid", "category"]], on="uid", how="left",
    )

    if neg_preds.empty:
        return None

    def _fp_table(preds: DataFrame) -> DataFrame:
        rows = []
        for model in preds["model"].unique():
            mp = preds[preds["model"] == model]
            n_total = len(mp)
            n_fp = mp["prediction"].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False,
            ).sum()
            rows.append({
                "Model": model,
                "Samples": n_total,
                "False Positives": int(n_fp),
                "FP Rate": n_fp / n_total if n_total > 0 else 0,
                "Precision": 1 - (n_fp / n_total) if n_total > 0 else 0,
            })
        result = DataFrame.from_records(rows)
        return result.sort_values("Precision", ascending=False)

    tables = {"all": _fp_table(neg_preds)}
    for cat in ["negative", "hard_negative"]:
        subset = neg_preds[neg_preds["category"] == cat]
        if not subset.empty:
            tables[cat] = _fp_table(subset)
    return tables


def _build_defense_delta() -> DataFrame | None:
    """Build a table showing how defense affects each model's recall.

    Columns: Model, Base Recall, Shield Recall, Delta.
    """
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    from evaluation.report.config import model_sort_key

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)

    merged = predictions.merge(
        dataset[["uid", "category", "pii_spans"]], on="uid", how="left",
    )
    pos = merged[merged["category"] == "positive"]
    pos = pos.copy()
    pos["has_gt"] = pos["pii_spans"].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False,
    )
    pos["has_pred"] = pos["prediction"].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False,
    )
    pos["tp"] = pos["has_gt"] & pos["has_pred"]

    recall_by_model = pos.groupby("model")["tp"].mean()

    # Latency: median ms per sample (if available)
    has_latency = "latency_ms" in predictions.columns
    latency_by_model = {}
    if has_latency:
        lat = predictions.dropna(subset=["latency_ms"])
        if not lat.empty:
            latency_by_model = (
                lat.groupby("model")["latency_ms"].median().to_dict()
            )

    rows = []
    for model in recall_by_model.index:
        if model.endswith("-defend"):
            continue
        defend = f"{model}-defend"
        base_r = recall_by_model.get(model)
        defend_r = recall_by_model.get(defend)
        if base_r is None:
            continue
        row = {
            "Model": model,
            "Base Recall": base_r,
            "Shield Recall": defend_r if defend_r is not None else None,
            "Delta": (defend_r - base_r) if defend_r is not None else None,
        }
        if latency_by_model:
            row["Latency (ms)"] = latency_by_model.get(model)
        rows.append(row)

    if not rows:
        return None

    result = DataFrame.from_records(rows)
    result["_ord"] = result["Model"].apply(model_sort_key)
    return result.sort_values("_ord").drop(columns=["_ord"])


def _build_fp_samples() -> str | None:
    """Build an HTML section showing which hard negative texts trigger FPs.

    Groups by model, shows the actual text of each false positive.
    """
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)
    predictions = predictions[predictions["model"] != "pii-shield"]

    hn = dataset[dataset["category"] == "hard_negative"]
    if hn.empty:
        return None

    hn_preds = predictions[predictions["uid"].isin(hn["uid"])].merge(
        hn[["uid", "llm_input"]], on="uid", how="left",
    )
    hn_preds["is_fp"] = hn_preds["prediction"].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False,
    )
    fps = hn_preds[hn_preds["is_fp"]]
    if fps.empty:
        return None

    html_parts = []
    for model in sort_models(fps["model"].unique()):
        model_fps = fps[fps["model"] == model]
        items = []
        for _, row in model_fps.iterrows():
            text = row["llm_input"]
            if isinstance(text, str):
                escaped = (
                    text[:120].replace("&", "&amp;")
                    .replace("<", "&lt;").replace(">", "&gt;")
                )
                items.append(
                    f'<li style="margin-bottom:0.3rem;font-size:0.8rem">'
                    f'<code style="color:var(--text-muted)">#{row["uid"]}</code> '
                    f'{escaped}{"..." if len(text) > 120 else ""}</li>'
                )
        html_parts.append(
            f'<div style="margin-bottom:1rem">'
            f'<strong style="font-size:0.85rem">'
            f'{display_name(model)}</strong>'
            f' <span style="color:var(--text-muted);font-size:0.78rem">'
            f'({len(model_fps)} FPs)</span>'
            f'<ul style="margin:0.3rem 0 0 1rem;padding:0">'
            f'{"".join(items)}</ul></div>'
        )

    return "".join(html_parts)


def _build_pii_type_analysis() -> dict[str, DataFrame] | None:
    """Compute per-PII-type Recall by attack, grouped per model.

    Returns a dict mapping model name to a DataFrame with columns:
    PII Type, Attack, Recall.
    """
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)

    positive = dataset[dataset["category"] == "positive"]
    if positive.empty:
        return None

    pos_merged = predictions.merge(
        positive[["uid", "pii_spans", "attack_target"]], on="uid", how="inner",
    )

    def _attack_label(target):
        if not isinstance(target, dict):
            return "Clean Positives"
        parts = (target.get("pii", []) or []) + (target.get("context", []) or [])
        if not parts:
            return "Clean Positives"
        return " + ".join(display_name(p) for p in parts)

    pos_merged["attack_label"] = pos_merged["attack_target"].apply(_attack_label)

    # Order attacks: clean first, then by number of techniques
    def _attack_sort_key(label):
        if label == "Clean Positives":
            return (0, "")
        parts = label.split(" + ")
        return (len(parts), label)

    all_attacks = sorted(pos_merged["attack_label"].unique(), key=_attack_sort_key)

    per_model = {}
    for model in pos_merged["model"].unique():
        model_data = pos_merged[pos_merged["model"] == model]
        rows = []

        for attack in all_attacks:
            if attack not in model_data["attack_label"].values:
                continue
            subset = model_data[model_data["attack_label"] == attack]
            type_tp = {}
            type_fn = {}

            for _, row in subset.iterrows():
                gt_spans = row["pii_spans"] if isinstance(row["pii_spans"], list) else []
                pred_spans = row["prediction"] if isinstance(row["prediction"], list) else []

                # Guard models return value=None — treat any
                # non-empty prediction as a binary detection hit.
                has_any_pred = len(pred_spans) > 0
                pred_values = [
                    s.get("value") for s in pred_spans if isinstance(s, dict)
                ]
                is_binary = all(v is None for v in pred_values)
                matched = set()

                for s in gt_spans:
                    if not isinstance(s, dict):
                        continue
                    pii_type = s.get("type", "unknown")
                    if is_binary:
                        found = has_any_pred
                    else:
                        gt_val = s.get("value", "")
                        found = False
                        for pi, pv in enumerate(pred_values):
                            if (pi not in matched and pv
                                    and (gt_val in pv or pv in gt_val)):
                                found = True
                                matched.add(pi)
                                break
                    bucket = type_tp if found else type_fn
                    bucket[pii_type] = bucket.get(pii_type, 0) + 1

            for pii_type in sorted(set(type_tp) | set(type_fn)):
                tp = type_tp.get(pii_type, 0)
                fn = type_fn.get(pii_type, 0)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                rows.append({
                    "PII Type": pii_type,
                    "Attack": attack,
                    "Recall": round(recall, 4),
                })

        if rows:
            per_model[model] = DataFrame.from_records(rows)

    return per_model if per_model else None


def _dataset_label(row):
    """Categorise a sample as baseline or adversarial."""
    if row["category"] in ("negative", "hard_negative"):
        return "baseline"
    target = row["attack_target"]
    has_attack = (
        isinstance(target, dict)
        and (
            bool(target.get("pii"))
            or bool(target.get("context"))
        )
    )
    return "fuzzy_adv" if has_attack else "baseline"


def _build_perplexity_charts() -> dict[str, str] | None:
    """Build perplexity charts for every model that has data.

    Returns dict mapping model name to chart HTML (static +
    interactive), or None when no perplexity data exists.
    """
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)

    if "perplexity" not in predictions.columns:
        return None

    has_perp = predictions.dropna(subset=["perplexity"])
    if has_perp.empty:
        return None

    chosen = Config.PERPLEXITY_THRESHOLD
    charts: dict[str, str] = {}

    for model in sorted(has_perp["model"].unique()):
        model_preds = has_perp[has_perp["model"] == model]
        merged = model_preds.merge(
            dataset[["uid", "pii_spans", "category", "attack_target"]],
            on="uid", how="left",
        )
        merged = merged.dropna(subset=["perplexity"])
        if merged.empty:
            continue

        # Adaptive threshold range based on actual perplexity values
        ppl_min = merged["perplexity"].min()
        ppl_max = merged["perplexity"].max()
        ppl_range = ppl_max - ppl_min
        if ppl_range < 0.001:
            # Very narrow range (e.g. GPT-4o-mini near 1.0)
            thresholds = np.arange(
                ppl_min, ppl_max + 0.000001, 0.00000001,
            )
        else:
            # Wider range (e.g. Qwen Guard 1.0-2.1)
            thresholds = np.linspace(
                max(1.0, ppl_min - 0.1),
                ppl_max + 0.1,
                500,
            )

        merged["y_true"] = merged["pii_spans"].apply(
            lambda x: (
                len(x) > 0 if isinstance(x, list)
                else False
            ),
        )
        merged["model"] = model
        merged["dataset"] = merged.apply(
            _dataset_label, axis=1,
        )

        fig = plot_threshold_sweep(
            df=merged, thresholds=thresholds,
            chosen_threshold=chosen,
        )
        pj = plot_threshold_sweep_plotly(
            df=merged, thresholds=thresholds,
            chosen_threshold=chosen,
        )
        charts[model] = _chart_panel(
            fig_to_base64(fig), pj,
        )

    return charts if charts else None


def _build_inspector() -> str | None:
    """Build the sample inspector section.

    Embeds sample data as JSON and renders an interactive
    viewer where users can pick a model, filter by verdict
    (TP/FP/TN/FN) and category, and see highlighted PII
    spans on the original input text.
    """
    if not PREDICTIONS_PATH.exists() or not DATASET_PATH.exists():
        return None

    dataset = _load_report_dataset()
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)
    predictions = predictions[predictions["model"] != "pii-shield"]

    if predictions.empty:
        return None

    def _segment(row):
        cat = row["category"]
        if cat == "negative":
            return "Negatives"
        if cat == "hard_negative":
            return "Hard Negatives"
        t = row["attack_target"]
        if not isinstance(t, dict):
            return "Positives"
        has_pii = bool(t.get("pii"))
        has_ctx = bool(t.get("context"))
        if has_pii and not has_ctx:
            return "Direct Attack"
        if has_ctx:
            return "Direct + Indirect"
        return "Positives"

    dataset["segment"] = dataset.apply(_segment, axis=1)

    # Build samples dict (shared across models)
    from data_manipulation.defenses.preprocess import (
        defensive_preprocess,
        light_defensive_preprocess,
    )
    from evaluation.report.span_locator import locate_span_in_defended

    samples = {}
    for _, r in dataset.iterrows():
        uid = int(r["uid"])
        gt = r["pii_spans"]
        if not isinstance(gt, list):
            gt = []
        gt_clean = [
            {"v": s.get("value", ""),
             "s": s.get("start"),
             "e": s.get("end"),
             "t": s.get("type", "")}
            for s in gt if isinstance(s, dict)
        ]
        raw = r["llm_input"] or ""
        defended_full = defensive_preprocess(raw)
        defended_light = light_defensive_preprocess(raw)

        def _build_dg(defended_text):
            spans = []
            for s in gt:
                if not isinstance(s, dict):
                    continue
                val = s.get("value", "")
                typ = s.get("type", "")
                orig_s = s.get("start")
                orig_e = s.get("end")
                raw_fragment = (
                    raw[orig_s:orig_e]
                    if orig_s is not None and orig_e is not None
                    else val
                )
                start, end = locate_span_in_defended(
                    raw_fragment, defended_text, original_value=val,
                )
                matched = (
                    defended_text[start:end]
                    if start is not None else val
                )
                spans.append(
                    {"v": matched, "s": start, "e": end, "t": typ},
                )
            return spans

        samples[uid] = {
            "x": raw,
            "d": defended_full,
            "dl": defended_light,
            "dg": _build_dg(defended_full),
            "dlg": _build_dg(defended_light),
            "g": gt_clean,
            "c": r["segment"],
        }

    # Build per-model predictions
    model_preds = {}
    for model in sort_models(predictions["model"].unique()):
        mp = predictions[predictions["model"] == model]
        preds = {}
        for _, r in mp.iterrows():
            uid = int(r["uid"])
            pred = r["prediction"]
            if not isinstance(pred, list):
                pred = []
            pred_clean = [
                {"v": s.get("value", ""),
                 "s": s.get("start"),
                 "e": s.get("end"),
                 "t": s.get("type", "")}
                for s in pred if isinstance(s, dict)
            ]
            has_gt = (
                uid in samples
                and len(samples[uid]["g"]) > 0
            )
            has_pred = len(pred_clean) > 0
            if has_gt and has_pred:
                verdict = "TP"
            elif not has_gt and has_pred:
                verdict = "FP"
            elif has_gt and not has_pred:
                verdict = "FN"
            else:
                verdict = "TN"
            preds[uid] = {
                "p": pred_clean,
                "r": verdict,
            }
        model_preds[model] = preds

    from pipelines import _SLM_MODELS

    inspector_data = json.dumps({
        "s": samples,
        "m": {
            m: {
                "d": display_name(m),
                "p": p,
                "light": m.removesuffix("-defend") in _SLM_MODELS,
            }
            for m, p in model_preds.items()
        },
    }, ensure_ascii=False)

    return inspector_data


def _build_comparison_charts(report_data: dict) -> str | None:
    """Build grouped bar charts with metric toggle (F1 / Recall / Precision)."""
    from evaluation.visualizations.visualizations import grouped_bar_plotly

    metrics = ["F1", "Recall", "Precision"]
    sections_by_key = {
        "fuzzy": ("PII-Level Attacks", "fuzzy_techniques"),
        "adv": ("Content-Level Attacks", "adv_content_techniques"),
    }

    # Build charts grouped by metric — base & shield side by side
    metric_panels: dict[str, list[str]] = {m: [] for m in metrics}
    chart_idx = 0
    for key, (title, group_col) in sections_by_key.items():
        df = report_data.get(key)
        if df is None or df.empty:
            continue

        base_df = df[~df["Model"].str.endswith("-defend")]
        shield_df = df[df["Model"].str.endswith("-defend")].copy()
        shield_df["Model"] = shield_df["Model"].str.removesuffix("-defend")

        for metric in metrics:
            halves = []
            for split_label, split_df in [
                ("Base Models", base_df),
                ("Shield Models", shield_df),
            ]:
                if split_df.empty:
                    continue
                pid = f"plotly-comp-{chart_idx}"
                chart_idx += 1
                pj = grouped_bar_plotly(split_df, metric, group_col)
                halves.append(
                    f'<div style="flex:1;min-width:0">'
                    f'<h4 style="margin:0 0 0.3rem;font-size:0.85rem;'
                    f'color:var(--text-muted)">{split_label}</h4>'
                    f'<div class="chart-wrap" data-plotly-id="{pid}">'
                    f'<div class="chart-interactive" id="{pid}" '
                    f'style="min-height:350px"></div>'
                    f'<script type="application/json" class="plotly-spec" '
                    f'data-target="{pid}">{pj}</script></div></div>'
                )
            if halves:
                metric_panels[metric].append(
                    f'<h3 style="margin:1.5rem 0 0.3rem;font-size:0.9rem">'
                    f'{title}</h3>'
                    f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap">'
                    f'{"".join(halves)}</div>'
                )

    if not any(metric_panels.values()):
        return None

    # Metric toggle buttons
    btns = []
    for m in metrics:
        active = " active" if m == metrics[0] else ""
        btns.append(
            f'<button class="tab-btn comp-metric-btn{active}" '
            f'data-comp-metric="{m}">{m}</button>'
        )
    bar = (
        f'<div class="view-bar" style="margin-bottom:0.8rem">'
        f'{"".join(btns)}</div>'
    )

    # Metric panels
    panels = []
    for m in metrics:
        display = "block" if m == metrics[0] else "none"
        panels.append(
            f'<div class="comp-panel" data-comp-metric="{m}" '
            f'style="display:{display}">{"".join(metric_panels[m])}</div>'
        )

    return bar + "".join(panels)


def generate_report(output_path: Path = None, open_browser: bool = True) -> Path:
    apply_style()

    if output_path is None:
        output_path = DATASET_PATH.parent / "report.html"

    report_data = _compute_report_data()
    dataset = _load_report_dataset()

    sections_html = []
    nav_items = []

    # 1. Overview
    if not dataset.empty:
        overview_html = _build_overview(dataset)
        leaderboard_html = _build_leaderboard() or ""
        if leaderboard_html:
            leaderboard_html = (
                '<h3 style="margin:1.5rem 0 0.5rem;font-size:0.95rem">'
                'Benchmarks</h3>'
                '<p class="section-desc">Binary detection rate across dataset segments. '
                'Recall measures how often PII is detected; Precision measures how '
                'often negative samples are correctly ignored.</p>'
                + leaderboard_html
            )
        shield_card_html = _build_shield_card() or ""
        sections_html.append(render_static_section(
            "overview", "Dataset Overview",
            overview_html + leaderboard_html + shield_card_html,
        ))
        nav_items.append('<a class="nav-pill" data-page="overview">Overview</a>')

    # 2. Consolidated Performance page — always show all 3 sub-tabs
    perf_subsections = [
        (section, report_data.get(section["key"]))
        for section in DATA_SECTIONS
    ]
    has_perf = any(df is not None and not df.empty for _, df in perf_subsections)

    if has_perf:
        sections_html.append(render_performance_page(perf_subsections))
        nav_items.append('<a class="nav-pill" data-page="performance">Performance</a>')

    # 3. False Positive Analysis
    fp_tables = _build_fp_analysis()
    if fp_tables:
        fp_col_order = ["Model", "Samples", "False Positives", "FP Rate", "Precision"]
        fp_pct = ["FP Rate", "Precision"]
        fp_desc = (
            '<p class="section-desc">False positive rate — a false positive '
            'is when the detector incorrectly flags clean text as containing PII.</p>'
        )
        fp_tab_labels = {
            "all": "All Negatives",
            "negative": "Negatives",
            "hard_negative": "Hard Negatives",
        }
        fp_tabs = []
        fp_panels = []
        for i, (key, label) in enumerate(fp_tab_labels.items()):
            df = fp_tables.get(key)
            if df is None or df.empty:
                continue
            active = " active" if i == 0 else ""
            display = "block" if i == 0 else "none"
            fp_tabs.append(
                f'<button class="tab-btn fp-tab{active}" '
                f'data-fp-target="{key}">{label}</button>'
            )
            fp_panels.append(
                f'<div class="fp-panel" data-fp-panel="{key}" '
                f'style="display:{display}">'
                f'{styled_table(df, col_order=fp_col_order, pct_cols=fp_pct)}'
                f'</div>'
            )
        fp_bar = (
            f'<div class="view-bar" style="margin-bottom:0.8rem">'
            f'{"".join(fp_tabs)}</div>'
        )
        fp_samples_html = _build_fp_samples() or ""
        if fp_samples_html:
            fp_samples_html = (
                '<h4 style="margin:1.5rem 0 0.5rem;font-size:0.88rem">'
                'Texts Triggering False Positives</h4>'
                + fp_samples_html
            )
        fp_content = fp_desc + fp_bar + "".join(fp_panels) + fp_samples_html
        sections_html.append(render_static_section(
            "fp-analysis", "False Positive Analysis", fp_content,
        ))
        nav_items.append('<a class="nav-pill" data-page="fp-analysis">FP Analysis</a>')

    # 4. Defense Effectiveness
    delta_df = _build_defense_delta()
    if delta_df is not None and not delta_df.empty:
        delta_desc = (
            '<p class="section-desc">'
            'How defensive preprocessing affects each model\'s recall. '
            'Positive delta means defense helps; negative means defense '
            'hurts. Pattern-based detectors benefit from aggressive '
            'normalization; SLMs use a lighter defense to preserve '
            'natural language context.</p>'
        )
        col_order = ["Model", "Base Recall", "Shield Recall", "Delta"]
        if "Latency (ms)" in delta_df.columns:
            col_order.append("Latency (ms)")
        delta_table = styled_table(
            delta_df,
            col_order=col_order,
            pct_cols=["Base Recall", "Shield Recall", "Delta"],
        )
        sections_html.append(render_static_section(
            "defense", "Defense Effectiveness", delta_desc + delta_table,
        ))
        nav_items.append(
            '<a class="nav-pill" data-page="defense">Defense</a>'
        )

    # 5. Model Comparison — grouped bar charts
    comparison_html = _build_comparison_charts(report_data)
    if comparison_html:
        comp_desc = (
            '<p class="section-desc">Grouped bar charts comparing model performance '
            'across attack categories. Each chart shows a single metric with bars '
            'grouped by model, split into base and shield variants.</p>'
        )
        sections_html.append(render_static_section(
            "comparison", "Model Comparison", comp_desc + comparison_html,
        ))
        nav_items.append(
            '<a class="nav-pill" data-page="comparison">Comparison</a>'
        )

    # 5. Per-PII-Type Analysis — heatmap per model (Recall × PII Type × Attack)
    pii_per_model = _build_pii_type_analysis()
    if pii_per_model:
        from evaluation.visualizations.heatmap import heatmap_plotly
        ordered_models = sort_models(pii_per_model.keys())
        model_tabs = []
        model_panels = []
        for midx, model in enumerate(ordered_models):
            mdf = pii_per_model[model]
            active = " active" if midx == 0 else ""
            display = "block" if midx == 0 else "none"
            safe_id = model.replace(".", "-")
            is_meta = model == "pii-shield"
            if is_meta:
                model_tabs.append('<span class="tab-separator"></span>')
            meta_cls = " meta-model" if is_meta else ""
            model_tabs.append(
                f'<button class="sub-tab{active}{meta_cls}"'
                f' data-sub="pii-{safe_id}">'
                f'{display_name(model)}</button>'
            )
            pid = f"plotly-pii-{safe_id}"
            pj = heatmap_plotly(mdf, "Recall", "Attack", cols_col="PII Type")
            chart = (
                f'<div class="chart-wrap" data-plotly-id="{pid}">'
                f'<div class="chart-interactive" id="{pid}"></div>'
                f'<script type="application/json" class="plotly-spec" '
                f'data-target="{pid}">{pj}</script></div>'
            )
            model_panels.append(
                f'<div class="sub-panel" data-sub="pii-{safe_id}" '
                f'style="display:{display}">{chart}</div>'
            )

        pii_desc = (
            '<p class="section-desc">Per-PII-type recall broken down by attack '
            'combination. Rows are attack configurations (ordered by complexity), '
            'columns are PII entity types. Select a model to view its heatmap.</p>'
        )
        pii_html = (
            f'{pii_desc}'
            f'<div class="sub-nav" style="margin-bottom:1rem">{"".join(model_tabs)}</div>'
            f'{"".join(model_panels)}'
        )
        sections_html.append(render_static_section(
            "pii-type", "Per-PII-Type Recall by Attack", pii_html,
        ))
        nav_items.append('<a class="nav-pill" data-page="pii-type">PII Types</a>')

    # 5. Perplexity — per-model charts with interactive toggle
    perplexity_charts = _build_perplexity_charts()
    if perplexity_charts:
        perplexity_desc = (
            '<p class="section-desc">'
            'Sensitivity analysis of the perplexity '
            'threshold. Shows how Baseline Precision, '
            'Baseline Recall, and Adversarial Recall '
            'change across threshold values. The dashed '
            'vertical line marks the chosen threshold '
            f'({Config.PERPLEXITY_THRESHOLD}).</p>'
        )

        if len(perplexity_charts) == 1:
            model_name = next(iter(perplexity_charts))
            perp_body = (
                f'{perplexity_desc}'
                f'{perplexity_charts[model_name]}'
            )
        else:
            tabs = []
            panels = []
            for midx, model in enumerate(
                sort_models(perplexity_charts.keys()),
            ):
                chart = perplexity_charts[model]
                active = " active" if midx == 0 else ""
                display = (
                    "block" if midx == 0 else "none"
                )
                safe_id = model.replace(".", "-")
                tabs.append(
                    f'<button class="sub-tab{active}" '
                    f'data-sub="perp-{safe_id}">'
                    f'{display_name(model)}</button>'
                )
                panels.append(
                    f'<div class="sub-panel" '
                    f'data-sub="perp-{safe_id}" '
                    f'style="display:{display}">'
                    f'{chart}</div>'
                )
            perp_body = (
                f'{perplexity_desc}'
                f'<div class="sub-nav" '
                f'style="margin-bottom:1rem">'
                f'{"".join(tabs)}</div>'
                f'{"".join(panels)}'
            )

        sections_html.append(render_static_section(
            "perplexity", "Perplexity Analysis",
            perp_body,
        ))
        nav_items.append(
            '<a class="nav-pill" '
            'data-page="perplexity">Perplexity</a>'
        )

    # 6. Inspector
    inspector_json = _build_inspector()
    if inspector_json:
        inspector_body = (
            '<p class="section-desc">'
            'Select a detector to inspect individual '
            'samples grouped by category. '
            'Expected PII is highlighted in '
            '<span style="background:rgba(69,117,180,0.25)'
            ';padding:1px 2px;border-radius:2px">'
            'blue</span>, '
            'detected spans in '
            '<span style="background:rgba(244,109,67,0.25)'
            ';padding:1px 2px;border-radius:2px">'
            'orange</span>, '
            'correctly matched in '
            '<span style="background:rgba(80,180,80,0.25)'
            ';padding:1px 2px;border-radius:2px">'
            'green</span>.</p>'
            '<div id="inspector-model-tabs" '
            'class="sub-nav" '
            'style="margin-bottom:0.8rem"></div>'
            '<div style="margin-bottom:0.8rem">'
            '<input type="text" id="inspector-search" '
            'placeholder='
            '"Search by uid, text, or PII value..." '
            'style="width:100%;padding:0.5rem 0.8rem;'
            'border:1px solid var(--border);'
            'border-radius:var(--radius);'
            'background:var(--bg);color:var(--text);'
            'font-size:0.82rem;'
            'font-family:var(--font-body);'
            'outline:none" /></div>'
            '<div id="inspector-sections"></div>'
            f'<script type="application/json" '
            f'id="inspector-data">'
            f'{inspector_json}</script>'
        )
        sections_html.append(render_static_section(
            "inspector", "Sample Inspector",
            inspector_body,
        ))
        nav_items.append(
            '<a class="nav-pill" '
            'data-page="inspector">Inspector</a>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PII Detection \u2014 Evaluation Report</title>
{PLOTLY_CDN}
<style>{CSS}</style>
</head>
<body>
<div class="topbar">
    <span class="topbar-title">PII Detection Report</span>
    <div class="topbar-center">{"".join(nav_items)}</div>
    <div class="topbar-right">
        <button class="theme-toggle" id="theme-toggle">&#9789; Dark</button>
    </div>
</div>
{"".join(sections_html)}
<div class="footer">PII Under Attack \u2014 Evaluation Dashboard</div>
<script>{JS}</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())

    return output_path
