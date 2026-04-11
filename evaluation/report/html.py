"""HTML rendering components for the evaluation report: styled tables, section
layouts, CSS theming, JavaScript interactivity (navigation, sorting, Plotly
chart initialization), and chart panel wrappers.
"""
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame

from evaluation.report.config import display_name
from evaluation.visualizations.heatmap import heatmap, heatmap_plotly
from evaluation.visualizations.line_chart import line_chart, line_plotly
from evaluation.visualizations.radar import radar_chart, radar_plotly
from evaluation.visualizations.style import fig_to_base64


# ── Table rendering ─────────────────────────────────────────────────────────

def styled_table(df: DataFrame,
                 col_order: list[str] = None,
                 pct_cols: list[str] = None) -> str:
    """Render a DataFrame as a styled HTML table.

    pct_cols: columns to format as percentages. If None, all float columns
              are formatted as percentages. Integer columns are left as-is.
    """
    display_df = df.copy()

    if col_order:
        display_df = display_df[[c for c in col_order if c in display_df.columns]]
    else:
        str_cols = [c for c in display_df.columns if display_df[c].dtype == "object"]
        num_cols = display_df.select_dtypes("number").columns.tolist()
        display_df = display_df[str_cols + num_cols]

    for col in display_df.columns:
        if display_df[col].dtype == "object":
            display_df[col] = display_df[col].map(
                lambda v: display_name(v) if isinstance(v, str) else v
            )
    display_df = display_df.rename(columns={c: display_name(c) for c in display_df.columns})

    # Determine which columns get percentage formatting
    if pct_cols is not None:
        pct_display = [display_name(c) for c in pct_cols if display_name(c) in display_df.columns]
    else:
        pct_display = display_df.select_dtypes("float").columns.tolist()

    int_cols = display_df.select_dtypes("integer").columns.tolist()
    numeric_cols = display_df.select_dtypes("number").columns.tolist()
    str_cols = [c for c in display_df.columns if c not in numeric_cols]

    css = [
        {"selector": "", "props": [
            ("border-collapse", "collapse"), ("width", "100%"),
            ("font-size", "13px"), ("table-layout", "auto"),
        ]},
        {"selector": "thead th", "props": [
            ("background", "#f8f9fa"), ("color", "#333"),
            ("font-weight", "600"), ("padding", "10px 10px"),
            ("border-bottom", "2px solid #dee2e6"),
            ("text-align", "center"), ("font-size", "12px"),
            ("cursor", "pointer"), ("user-select", "none"),
        ]},
        {"selector": "tbody td", "props": [
            ("padding", "6px 10px"), ("text-align", "center"),
            ("border-bottom", "1px solid #eee"),
        ]},
        {"selector": "tbody tr:nth-child(even)", "props": [
            ("background", "#fafbfc"),
        ]},
        {"selector": "tbody tr:hover td", "props": [
            ("background", "#e3edf7 !important"),
        ]},
        {"selector": "tbody tr:last-child td", "props": [
            ("border-bottom", "2px solid #dee2e6"),
        ]},
        {"selector": "thead tr:nth-child(2)", "props": [("display", "none")]},
    ]
    cmap = LinearSegmentedColormap.from_list(
        "tbl", ["#ffffff", "#d6e8f5", "#8cb8d8", "#3a7ebf", "#1a4e7e"]
    )
    if pct_display:
        display_df[pct_display] = display_df[pct_display].fillna(0)
    styler = display_df.style.set_table_styles(css).hide(axis="index")
    if pct_display:
        styler = styler.format("{:.1%}", subset=pct_display)
        styler = styler.background_gradient(
            cmap=cmap, subset=pct_display, vmin=0, vmax=1,
        )
        styler = styler.map(
            lambda v: "color: #fff; font-weight: 700"
            if isinstance(v, (int, float)) and v > 0.55
            else "color: #1a1a1a; font-weight: 600",
            subset=pct_display,
        )
    if int_cols:
        styler = styler.format("{:,}", subset=int_cols)
    for col in str_cols:
        styler = styler.set_properties(
            subset=[col], **{"text-align": "left", "font-weight": "500",
                             "max-width": "180px", "overflow": "hidden",
                             "text-overflow": "ellipsis"},
        )
    return styler.to_html()


# ── Section config ───────────────────────────────────────────────────────────

_PERF_METRICS = ["F1", "Recall", "Precision"]

DATA_SECTIONS = [
    {"id": "fuzzy", "title": "PII-Level Attacks",
     "key": "fuzzy",
     "index_col": "fuzzy_techniques",
     "desc": "Impact of PII-level adversarial "
             "transformations (homoglyph, chunking, "
             "emojify, etc.) on detector recall."},
    {"id": "adversarial",
     "title": "Content-Level Attacks",
     "key": "adv",
     "index_col": "adv_content_techniques",
     "desc": "Impact of content-level attacks "
             "(supportive context, prompt injection, "
             "affixes) combined with PII-level "
             "fuzzing."},
]

_VIEW_TYPES = ["table", "heatmap", "radar", "line"]
_VIEW_LABELS = {
    "table": "Table", "heatmap": "Heatmap", "radar": "Radar", "line": "Line Chart",
}
_VIEW_ICONS = {
    "table": "&#9776;", "heatmap": "&#9638;", "radar": "&#9678;", "line": "&#9413;",
}

_COUNTER = [0]


def _uid():
    _COUNTER[0] += 1
    return f"plotly-{_COUNTER[0]}"


def _chart_panel(static_img_b64: str, plotly_json: str) -> str:
    """Wrap a chart in a container with both static and interactive versions."""
    pid = _uid()
    return (
        f'<div class="chart-wrap" data-plotly-id="{pid}">'
        f'<div class="chart-interactive" id="{pid}"></div>'
        f'<div class="chart-static" style="display:none">'
        f'<img data-src="{static_img_b64}" alt="chart"></div>'
        f'<script type="application/json" class="plotly-spec" '
        f'data-target="{pid}">{plotly_json}</script>'
        f'</div>'
    )


# ── Render consolidated performance section ──────────────────────────────────

def _strip_defend(df: DataFrame) -> DataFrame:
    """Remove '-defend' suffix from Model column."""
    out = df.copy()
    out["Model"] = out["Model"].str.removesuffix(
        "-defend",
    )
    return out


def _build_views_for_split(
    df: DataFrame, metric: str, index_col: str | None,
) -> dict[str, str]:
    """Build table/heatmap/radar/line views for a df."""
    # Table: keep only Model, index_col, and the selected metric; sort by it
    table_cols = ["Model"]
    if index_col and index_col in df.columns:
        table_cols.append(index_col)
    table_cols.append(metric)
    table_df = df[[c for c in table_cols if c in df.columns]].copy()
    if metric in table_df.columns:
        table_df = table_df.sort_values(metric, ascending=False)
    views = {"table": styled_table(table_df)}

    if index_col and index_col in df.columns:
        fig = heatmap(df, metric, index_col)
        pj = heatmap_plotly(df, metric, index_col)
        views["heatmap"] = _chart_panel(
            fig_to_base64(fig), pj,
        )

        groups = df[index_col].unique()
        if len(groups) >= 3:
            fig = radar_chart(df, metric, index_col)
            pj = radar_plotly(df, metric, index_col)
            views["radar"] = _chart_panel(
                fig_to_base64(fig), pj,
            )

        fig = line_chart(df, metric, index_col)
        pj = line_plotly(df, metric, index_col)
        views["line"] = _chart_panel(
            fig_to_base64(fig), pj,
        )

    return views


def _render_perf_section_body(
    section: dict, df: DataFrame | None,
) -> str:
    """Build per-view content for each metric, split into Base/Shield."""
    desc = section.get("desc", "")
    desc_html = (
        f'<p class="section-desc">{desc}</p>'
        if desc else ""
    )

    if df is None or df.empty:
        placeholder = (
            f'{desc_html}'
            f'<p style="color:var(--text-muted)">'
            f'No data available.</p>'
        )
        return "".join(
            f'<div class="perf-view" '
            f'data-perf-view="{v}" '
            f'style="display:'
            f'{"block" if v == "table" else "none"}">'
            f'{placeholder}</div>'
            for v in _VIEW_TYPES
        )

    index_col = section["index_col"]
    base_df = df[~df["Model"].str.endswith("-defend")]
    shield_df = _strip_defend(
        df[df["Model"].str.endswith("-defend")],
    )

    _BASE_LABEL = (
        '<h4 style="margin:0.5rem 0 0.4rem;'
        'font-size:0.85rem;'
        'color:var(--text-muted)">Base Models</h4>'
    )
    _SHIELD_LABEL = (
        '<h4 style="margin:0.5rem 0 0.4rem;'
        'font-size:0.85rem;'
        'color:var(--text-muted)">Shield Models</h4>'
    )

    panels = []
    for metric in _PERF_METRICS:
        base_views = _build_views_for_split(
            base_df, metric, index_col,
        )
        shield_views = _build_views_for_split(
            shield_df, metric, index_col,
        )

        available_types = set()
        for vtype in _VIEW_TYPES:
            if (base_views.get(vtype)
                    or shield_views.get(vtype)):
                available_types.add(vtype)

        for vtype in _VIEW_TYPES:
            if vtype not in available_types:
                continue
            is_default = (
                vtype == "table"
                and metric == _PERF_METRICS[0]
            )
            display = "block" if is_default else "none"

            bv = base_views.get(vtype)
            sv = shield_views.get(vtype)

            if vtype == "table":
                # Tables stacked vertically
                parts = [desc_html]
                if bv:
                    parts.append(_BASE_LABEL + bv)
                if sv:
                    parts.append(_SHIELD_LABEL + sv)
                content = "".join(parts)
            else:
                # Charts side by side
                halves = []
                if bv:
                    halves.append(
                        f'<div style="flex:1;min-width:0">'
                        f'{_BASE_LABEL}{bv}</div>'
                    )
                if sv:
                    halves.append(
                        f'<div style="flex:1;min-width:0">'
                        f'{_SHIELD_LABEL}{sv}</div>'
                    )
                content = (
                    desc_html
                    + f'<div style="display:flex;gap:1.5rem;'
                    f'flex-wrap:wrap">{"".join(halves)}</div>'
                )

            panels.append(
                f'<div class="perf-view" '
                f'data-perf-view="{vtype}" '
                f'data-perf-metric="{metric}" '
                f'style="display:{display}">'
                f'{content}</div>'
            )
    return "".join(panels)


def render_performance_page(
    subsections: list[tuple[dict, DataFrame | None]],
) -> str:
    """Render performance page with shared view switcher."""
    if not subsections:
        return ""

    toggle = (
        '<label class="toggle-label">'
        '<input type="checkbox" id="interactive-toggle"'
        ' checked> Interactive charts</label>'
    )

    # Metric toggle
    metric_btns = []
    for m in _PERF_METRICS:
        active = " active" if m == _PERF_METRICS[0] else ""
        metric_btns.append(
            f'<button class="tab-btn perf-metric-btn'
            f'{active}" data-perf-metric="{m}">'
            f'{m}</button>'
        )
    metric_bar = (
        f'<div class="view-bar" '
        f'style="margin-bottom:0.8rem">'
        f'{"".join(metric_btns)}</div>'
    )

    # Shared view bar
    view_btns = []
    for vtype in _VIEW_TYPES:
        active = " active" if vtype == "table" else ""
        view_btns.append(
            f'<button class="tab-btn perf-view-btn'
            f'{active}" data-perf-target="{vtype}">'
            f'{_VIEW_ICONS[vtype]} '
            f'{_VIEW_LABELS[vtype]}</button>'
        )
    view_bar = (
        f'<div class="view-bar">'
        f'{"".join(view_btns)}</div>'
    )

    cards = []
    for section, df in subsections:
        body = _render_perf_section_body(section, df)
        cards.append(
            f'<div class="card" '
            f'style="margin-bottom:1.5rem">'
            f'<div class="card-header">'
            f'<h2>{section["title"]}</h2></div>'
            f'<div class="card-body">{body}</div>'
            f'</div>'
        )

    return (
        f'<div class="page-section" '
        f'data-page="performance">'
        f'<div style="margin-bottom:1rem">'
        f'<div style="display:flex;justify-content:'
        f'space-between;align-items:center">'
        f'<h2 style="font-family:var(--font-heading);'
        f'font-size:1.1rem;font-weight:600">'
        f'Detection Performance</h2>'
        f'{view_bar}</div>'
        f'<div style="display:flex;justify-content:'
        f'space-between;align-items:center;'
        f'margin-top:0.4rem">'
        f'{metric_bar}{toggle}</div></div>'
        f'{"".join(cards)}'
        f'</div>'
    )


def render_static_section(section_id: str, title: str, content: str) -> str:
    return (
        f'<div class="page-section" data-page="{section_id}">'
        f'<div class="card">'
        f'<div class="card-header"><h2>{title}</h2></div>'
        f'<div class="card-body">{content}</div>'
        f'</div></div>'
    )


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
:root {
    --bg: #f5f5f0; --card-bg: #ffffff; --text: #1a1a1a; --text-muted: #666666;
    --border: #d4d4d4; --accent: #4575b4; --accent-light: #e8f0fa;
    --radius: 6px;
    --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 2px 8px rgba(0,0,0,0.03);
    --font-body: 'Inter', -apple-system, 'Helvetica Neue', sans-serif;
    --font-heading: 'Source Serif 4', 'Times New Roman', Times, serif;
}
[data-theme="dark"] {
    --bg: #0f0f0f; --card-bg: #1a1a1a; --text: #e8e8e8; --text-muted: #999999;
    --border: #333333; --accent: #74add1; --accent-light: #1e2d3d;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 2px 8px rgba(0,0,0,0.2);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: var(--font-body); background: var(--bg); color: var(--text);
    max-width: 1400px; margin: 0 auto; padding: 0 2rem 2rem;
    transition: background 0.3s, color 0.3s; line-height: 1.5;
}

/* ── Topbar ── */
.topbar {
    position: sticky; top: 0; z-index: 100;
    background: var(--bg); padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 0.6rem; margin-bottom: 1.5rem;
}
.topbar-title {
    font-family: var(--font-heading); font-size: 1.1rem; font-weight: 600;
    white-space: nowrap;
}
.topbar-center { display: flex; gap: 0.3rem; flex-wrap: wrap; align-items: center; }
.topbar-right { display: flex; gap: 0.5rem; align-items: center; }

/* ── Nav pills ── */
.nav-pill {
    padding: 0.4rem 0.85rem; border-radius: 4px;
    border: 1px solid var(--border); background: var(--card-bg);
    color: var(--text-muted); cursor: pointer;
    font-size: 0.8rem; font-weight: 500; transition: all 0.15s;
    text-decoration: none;
}
.nav-pill:hover { border-color: var(--accent); color: var(--accent); }
.nav-pill.active { background: var(--accent); color: white; border-color: var(--accent); }

.theme-toggle {
    padding: 0.35rem 0.6rem; border-radius: 4px; border: 1px solid var(--border);
    background: var(--card-bg); color: var(--text-muted); cursor: pointer;
    font-size: 0.78rem; transition: all 0.15s;
}
.theme-toggle:hover { border-color: var(--accent); color: var(--accent); }

/* ── Pages ── */
.page-section { display: none; animation: fadeIn 0.2s ease; }
.page-section.active { display: block; }

/* ── Cards ── */
.card {
    background: var(--card-bg); border-radius: var(--radius);
    box-shadow: var(--shadow); margin-bottom: 1.5rem;
    border: 1px solid var(--border);
    transition: background 0.3s, border-color 0.3s;
}
.card-header {
    padding: 0.8rem 1.2rem; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 0.5rem; background: var(--card-bg);
    border-radius: var(--radius) var(--radius) 0 0;
}
.card-header h2 { font-family: var(--font-heading); font-size: 1.05rem; font-weight: 600; }
.card-body { padding: 1.2rem; overflow-x: auto; }
.card-body img { max-width: 100%; height: auto; border-radius: 4px; }

/* ── Sub-navigation (within performance page) ── */
.sub-nav { display: flex; gap: 0.25rem; }
.sub-tab {
    padding: 0.3rem 0.75rem; border-radius: 4px; border: 1px solid var(--border);
    background: transparent; color: var(--text-muted); cursor: pointer;
    font-size: 0.78rem; font-weight: 500; transition: all 0.15s;
    font-family: var(--font-body);
}
.sub-tab:hover { border-color: var(--accent); color: var(--accent); }
.sub-tab.active {
    background: var(--accent); color: white; border-color: var(--accent);
}
.sub-nav .tab-separator {
    width: 1px; align-self: stretch; background: var(--border); margin: 0 0.35rem;
}
.sub-tab.meta-model {
    border-style: dashed; position: relative;
    border-color: #4a148c; color: #4a148c;
}
.sub-tab.meta-model::after {
    content: "CASCADE"; font-size: 0.5rem; font-weight: 700; letter-spacing: 0.04em;
    position: absolute; top: -0.45rem; right: 0.3rem;
    background: #4a148c; color: #fff; padding: 0 0.25rem; border-radius: 2px;
    line-height: 1.2;
}
.sub-tab.meta-model:hover { border-color: #4a148c; color: #4a148c; }
.sub-tab.meta-model.active {
    background: #4a148c; color: #fff; border-color: #4a148c; border-style: solid;
}
[data-theme="dark"] .sub-tab.meta-model {
    border-color: #b388ff; color: #b388ff;
}
[data-theme="dark"] .sub-tab.meta-model::after { background: #7c4dff; }
[data-theme="dark"] .sub-tab.meta-model:hover { border-color: #b388ff; color: #b388ff; }
[data-theme="dark"] .sub-tab.meta-model.active {
    background: #7c4dff; color: #fff; border-color: #7c4dff; border-style: solid;
}

/* ── View bar (table/heatmap/radar/line tabs) ── */
.view-bar {
    display: flex; gap: 0.25rem; margin-bottom: 1rem;
    padding-bottom: 0.6rem; border-bottom: 1px solid var(--border);
}
.tab-btn {
    padding: 0.3rem 0.7rem; border-radius: 4px; border: 1px solid var(--border);
    background: transparent; color: var(--text-muted); cursor: pointer;
    font-size: 0.75rem; font-weight: 500; transition: all 0.15s;
    font-family: var(--font-body);
}
.tab-btn:hover { border-color: var(--accent); color: var(--accent); }
.tab-btn.active {
    background: var(--accent-light); color: var(--accent);
    border-color: var(--accent); font-weight: 600;
}

.tab-panel { animation: fadeIn 0.15s ease; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* ── Interactive toggle ── */
.toggle-label {
    font-size: 0.78rem; color: var(--text-muted); cursor: pointer;
    display: flex; align-items: center; gap: 0.3rem; user-select: none;
}
.toggle-label input { cursor: pointer; }

/* ── Charts ── */
.chart-wrap { width: 100%; margin: 0.5rem 0; }
.chart-interactive { min-height: 400px; }
.radar-row { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
.radar-row .chart-wrap { flex: 1; min-width: 300px; max-width: 50%; }

/* ── Overview grid ── */
.overview-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
}
.stat-box {
    text-align: center; padding: 1.2rem 0.8rem;
    border: 1px solid var(--border); border-radius: var(--radius);
    background: var(--bg);
}
.stat-box .stat-value {
    font-size: 2rem; font-weight: 700; color: var(--accent);
    font-family: var(--font-heading); line-height: 1.2;
}
.stat-box .stat-label {
    font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem;
}

/* ── Section descriptions ── */
.section-desc {
    font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.8rem;
    line-height: 1.5;
}

/* ── Table ── */
.sort-arrow { font-size: 0.65em; color: #aaa; margin-left: 3px; }
th:hover .sort-arrow { color: var(--accent); }

/* ── Dark overrides ── */
[data-theme="dark"] .card-body table { color: var(--text); }
[data-theme="dark"] .card-body th {
    background: #252525 !important; color: #e0e0e0 !important; border-color: #444 !important;
}
[data-theme="dark"] .card-body td { border-color: #333 !important; }
[data-theme="dark"] .card-body tbody tr:nth-child(even) { background: #222 !important; }
[data-theme="dark"] .card-body tbody tr:hover td { background: #2a3a4a !important; }
[data-theme="dark"] .topbar { background: var(--bg); }
[data-theme="dark"] .card-header { background: var(--card-bg); }
[data-theme="dark"] .stat-box { background: var(--card-bg); }

/* ── Inspector ── */
.insp-sample {
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 0.8rem; margin-bottom: 0.5rem; background: var(--card-bg);
}
.insp-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.insp-badge {
    font-size: 0.68rem; font-weight: 600; padding: 2px 8px;
    border-radius: 3px; text-transform: uppercase; letter-spacing: 0.5px;
}
.insp-badge-tp { background: #d4edda; color: #155724; }
.insp-badge-fp { background: #f8d7da; color: #721c24; }
.insp-badge-fn { background: #fff3cd; color: #856404; }
.insp-badge-tn { background: #d6e8f5; color: #1a4e7e; }
[data-theme="dark"] .insp-badge-tp { background: #1e3a1e; color: #8fd19e; }
[data-theme="dark"] .insp-badge-fp { background: #3a1e1e; color: #e88e8e; }
[data-theme="dark"] .insp-badge-fn { background: #3a3520; color: #e8d48e; }
[data-theme="dark"] .insp-badge-tn { background: #1e2d3d; color: #8eb8d1; }
.insp-text {
    font-family: 'Courier New', monospace; font-size: 0.8rem;
    line-height: 1.7; white-space: pre-wrap; word-break: break-word;
    color: var(--text); padding: 0.5rem; background: var(--bg);
    border-radius: 4px; border: 1px solid var(--border);
}
.insp-gt { background: rgba(69,117,180,0.2); border-radius: 2px; padding: 0 1px; }
.insp-pred { background: rgba(244,109,67,0.2); border-radius: 2px; padding: 0 1px; }
.insp-both { background: rgba(80,180,80,0.25); border-radius: 2px; padding: 0 1px; }
.insp-section {
    border: 1px solid var(--border); border-radius: var(--radius);
    margin-bottom: 0.8rem; overflow: hidden;
}
.insp-section-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 1rem; cursor: pointer; user-select: none;
    background: var(--bg); transition: background 0.15s;
}
.insp-section-header:hover { background: var(--accent-light); }
.insp-section-title { font-weight: 600; font-size: 0.88rem; }
.insp-section-counts { font-size: 0.75rem; color: var(--text-muted); display: flex; gap: 0.8rem; }
.insp-section-body { padding: 0.6rem 1rem; display: none; }
.insp-section.open .insp-section-body { display: block; }
.insp-section-chevron { transition: transform 0.2s; font-size: 0.75rem; color: var(--text-muted); }
.insp-section.open .insp-section-chevron { transform: rotate(90deg); }
.insp-sub-header {
    font-size: 0.8rem; font-weight: 600; color: var(--text-muted);
    margin: 0.6rem 0 0.3rem; padding-bottom: 0.2rem;
    border-bottom: 1px solid var(--border);
}
.insp-meta { display: flex; flex-direction: column; gap: 0.2rem; margin-top: 0.35rem; font-size: 0.73rem; line-height: 1.4; }
.insp-meta-label { font-weight: 600; display: inline-block; min-width: 62px; }
.insp-meta-expected { color: rgba(69,117,180,0.9); }
.insp-meta-detected { color: rgba(244,109,67,0.9); }

.footer {
    text-align: center; padding: 1.2rem 0 0;
    color: var(--text-muted); font-size: 0.78rem;
    border-top: 1px solid var(--border); margin-top: 0.5rem;
}
"""

JS = r"""
document.addEventListener('DOMContentLoaded', function() {
    /* ── Plotly initialization ── */
    var plotlyInited = {};
    function initPlotly() {
        if (typeof Plotly === 'undefined') return;
        var scripts = document.querySelectorAll('.plotly-spec');
        for (var i = 0; i < scripts.length; i++) {
            var script = scripts[i];
            var targetId = script.getAttribute('data-target');
            if (plotlyInited[targetId]) continue;
            var el = document.getElementById(targetId);
            if (!el) continue;
            /* skip if hidden */
            var page = el.closest('.page-section');
            if (page && !page.classList.contains('active')) continue;
            var sub = el.closest('.sub-panel');
            if (sub && sub.style.display === 'none') continue;
            var tab = el.closest('.tab-panel');
            if (tab && tab.style.display === 'none') continue;
            var ci = el.closest('.chart-interactive');
            if (ci && ci.style.display === 'none') continue;
            try {
                var spec = JSON.parse(script.textContent);
                Plotly.newPlot(targetId, spec.data, spec.layout,
                    {responsive: true, displayModeBar: false});
                plotlyInited[targetId] = true;
            } catch(e) { console.warn('Plotly init failed:', targetId, e); }
        }
    }

    /* ── Page navigation ── */
    var pills = document.querySelectorAll('.nav-pill[data-page]');
    var pages = document.querySelectorAll('.page-section');

    function showPage(pageId) {
        for (var i = 0; i < pages.length; i++) pages[i].classList.remove('active');
        for (var i = 0; i < pills.length; i++) pills[i].classList.remove('active');
        var target = document.querySelector('.page-section[data-page="' + pageId + '"]');
        var pill = document.querySelector('.nav-pill[data-page="' + pageId + '"]');
        if (target) target.classList.add('active');
        if (pill) pill.classList.add('active');
        setTimeout(initPlotly, 100);
    }
    for (var i = 0; i < pills.length; i++) {
        pills[i].addEventListener('click', function(e) {
            e.preventDefault(); showPage(this.getAttribute('data-page'));
        });
    }
    if (pills.length > 0) showPage(pills[0].getAttribute('data-page'));

    /* ── Shield toggle (overview leaderboard) ── */
    var shieldBtn = document.getElementById('shield-toggle');
    if (shieldBtn) {
        var shieldOn = false;
        shieldBtn.addEventListener('click', function() {
            shieldOn = !shieldOn;
            var base = document.getElementById('leaderboard-base');
            var shield = document.getElementById('leaderboard-shield');
            var icon = document.getElementById('shield-path');
            var txt = document.getElementById('shield-text-top');
            var hint = document.getElementById('shield-hint');
            if (base) base.style.display = shieldOn ? 'none' : 'block';
            if (shield) shield.style.display = shieldOn ? 'block' : 'none';
            if (icon) { icon.setAttribute('fill', shieldOn ? 'var(--accent)' : 'none'); icon.setAttribute('stroke', shieldOn ? 'var(--accent)' : 'var(--border)'); }
            if (txt) txt.setAttribute('fill', shieldOn ? 'white' : 'var(--text-muted)');
            if (hint) { hint.textContent = shieldOn ? 'Prevention Applied' : 'Click to Apply'; hint.style.color = shieldOn ? 'var(--accent)' : 'var(--text-muted)'; }
            shieldBtn.style.transform = shieldOn ? 'scale(1.1)' : 'scale(1)';
        });
    }

    /* ── Sub-tabs (performance page) ── */
    var subTabs = document.querySelectorAll('.sub-tab');
    for (var i = 0; i < subTabs.length; i++) {
        subTabs[i].addEventListener('click', function() {
            var sub = this.getAttribute('data-sub');
            var card = this.closest('.card');
            var allSubs = card.querySelectorAll('.sub-tab');
            for (var j = 0; j < allSubs.length; j++) allSubs[j].classList.remove('active');
            this.classList.add('active');
            var panels = card.querySelectorAll('.sub-panel');
            for (var j = 0; j < panels.length; j++) panels[j].style.display = 'none';
            var panel = card.querySelector('.sub-panel[data-sub="' + sub + '"]');
            if (panel) { panel.style.display = 'block'; setTimeout(initPlotly, 100); }
        });
    }

    /* ── Performance metric + view switcher ── */
    function getActivePerfMetric() {
        var btn = document.querySelector('.perf-metric-btn.active');
        return btn ? btn.getAttribute('data-perf-metric') : 'F1';
    }
    function getActivePerfView() {
        var btn = document.querySelector('.perf-view-btn.active');
        return btn ? btn.getAttribute('data-perf-target') : 'table';
    }
    function syncPerfPanels() {
        var metric = getActivePerfMetric();
        var view = getActivePerfView();
        var allViews = document.querySelectorAll('.perf-view');
        for (var j = 0; j < allViews.length; j++) {
            var m = allViews[j].getAttribute('data-perf-metric') === metric;
            var v = allViews[j].getAttribute('data-perf-view') === view;
            allViews[j].style.display = (m && v) ? 'block' : 'none';
        }
        setTimeout(initPlotly, 100);
    }

    var perfMetricBtns = document.querySelectorAll('.perf-metric-btn');
    for (var i = 0; i < perfMetricBtns.length; i++) {
        perfMetricBtns[i].addEventListener('click', function() {
            for (var j = 0; j < perfMetricBtns.length; j++) perfMetricBtns[j].classList.remove('active');
            this.classList.add('active');
            syncPerfPanels();
        });
    }

    var perfBtns = document.querySelectorAll('.perf-view-btn');
    for (var i = 0; i < perfBtns.length; i++) {
        perfBtns[i].addEventListener('click', function() {
            for (var j = 0; j < perfBtns.length; j++) perfBtns[j].classList.remove('active');
            this.classList.add('active');
            syncPerfPanels();
        });
    }

    /* ── FP Analysis tabs ── */
    var fpBtns = document.querySelectorAll('.fp-tab');
    for (var i = 0; i < fpBtns.length; i++) {
        fpBtns[i].addEventListener('click', function() {
            var target = this.getAttribute('data-fp-target');
            for (var j = 0; j < fpBtns.length; j++) fpBtns[j].classList.remove('active');
            this.classList.add('active');
            var panels = document.querySelectorAll('.fp-panel');
            for (var j = 0; j < panels.length; j++) {
                panels[j].style.display = panels[j].getAttribute('data-fp-panel') === target ? 'block' : 'none';
            }
        });
    }

    /* ── Model Comparison metric toggle ── */
    var compBtns = document.querySelectorAll('.comp-metric-btn');
    for (var i = 0; i < compBtns.length; i++) {
        compBtns[i].addEventListener('click', function() {
            var metric = this.getAttribute('data-comp-metric');
            for (var j = 0; j < compBtns.length; j++) compBtns[j].classList.remove('active');
            this.classList.add('active');
            var panels = document.querySelectorAll('.comp-panel');
            for (var j = 0; j < panels.length; j++) {
                panels[j].style.display = panels[j].getAttribute('data-comp-metric') === metric ? 'block' : 'none';
            }
            setTimeout(initPlotly, 100);
        });
    }

    /* ── Theme toggle ── */
    var themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', function() {
            var html = document.documentElement;
            var next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            themeBtn.textContent = next === 'dark' ? '\u2600 Light' : '\u263D Dark';
        });
    }

    /* ── Interactive toggle ── */
    var toggle = document.getElementById('interactive-toggle');
    if (toggle) {
        toggle.addEventListener('change', function() {
            var interactive = toggle.checked;
            var wraps = document.querySelectorAll('.chart-wrap');
            for (var i = 0; i < wraps.length; i++) {
                var ci = wraps[i].querySelector('.chart-interactive');
                var cs = wraps[i].querySelector('.chart-static');
                if (ci) ci.style.display = interactive ? '' : 'none';
                if (cs) {
                    cs.style.display = interactive ? 'none' : '';
                    if (!interactive) {
                        var img = cs.querySelector('img[data-src]');
                        if (img && !img.src.startsWith('data:')) {
                            img.src = 'data:image/png;base64,' + img.getAttribute('data-src');
                        }
                    }
                }
            }
            if (interactive) setTimeout(initPlotly, 100);
        });
    }

    /* ── Table sorting ── */
    var tables = document.querySelectorAll('.card-body table');
    for (var t = 0; t < tables.length; t++) {
        (function(table) {
            var headers = table.querySelectorAll('thead th');
            for (var h = 0; h < headers.length; h++) {
                (function(th, colIdx) {
                    th.innerHTML += ' <span class="sort-arrow">\u2195</span>';
                    th.addEventListener('click', function() {
                        var tbody = table.querySelector('tbody');
                        var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
                        var arrow = th.querySelector('.sort-arrow');
                        var asc = arrow.textContent !== '\u2191';
                        for (var k = 0; k < headers.length; k++) {
                            var a = headers[k].querySelector('.sort-arrow');
                            if (a) a.textContent = '\u2195';
                        }
                        arrow.textContent = asc ? '\u2191' : '\u2193';
                        rows.sort(function(a, b) {
                            var aCell = a.cells[colIdx], bCell = b.cells[colIdx];
                            if (!aCell || !bCell) return 0;
                            var aText = aCell.textContent.replace('%','').trim();
                            var bText = bCell.textContent.replace('%','').trim();
                            var aNum = parseFloat(aText), bNum = parseFloat(bText);
                            if (!isNaN(aNum) && !isNaN(bNum))
                                return asc ? aNum - bNum : bNum - aNum;
                            return asc ? aText.localeCompare(bText) : bText.localeCompare(aText);
                        });
                        for (var k = 0; k < rows.length; k++) tbody.appendChild(rows[k]);
                    });
                })(headers[h], h);
            }
        })(tables[t]);
    }

    setTimeout(initPlotly, 300);

    /* ── Inspector ── */
    var inspData = document.getElementById('inspector-data');
    if (inspData) {
        var data = JSON.parse(inspData.textContent);
        var samples = data.s;
        var models = data.m;
        var modelKeys = Object.keys(models);
        var curModel = modelKeys[0] || '';

        var mtabs = document.getElementById('inspector-model-tabs');
        for (var mi = 0; mi < modelKeys.length; mi++) {
            (function(mk) {
                var btn = document.createElement('button');
                btn.className = 'sub-tab' + (mk === curModel ? ' active' : '');
                btn.textContent = models[mk].d;
                btn.addEventListener('click', function() {
                    curModel = mk;
                    var all = mtabs.querySelectorAll('.sub-tab');
                    for (var j = 0; j < all.length; j++) all[j].classList.remove('active');
                    btn.classList.add('active');
                    renderInspector();
                });
                mtabs.appendChild(btn);
            })(modelKeys[mi]);
        }

        function highlightText(text, gtSpans, predSpans) {
            var marks = new Array(text.length);
            for (var c = 0; c < text.length; c++) marks[c] = 0;
            for (var g = 0; g < gtSpans.length; g++) {
                var gs = gtSpans[g];
                if (gs.s != null && gs.e != null)
                    for (var c = gs.s; c < gs.e && c < text.length; c++) marks[c] |= 1;
            }
            for (var p = 0; p < predSpans.length; p++) {
                var ps = predSpans[p];
                if (ps.s != null && ps.e != null)
                    for (var c = ps.s; c < ps.e && c < text.length; c++) marks[c] |= 2;
            }
            var html = '', curMark = -1;
            var cls = {0:'', 1:'insp-gt', 2:'insp-pred', 3:'insp-both'};
            for (var c = 0; c < text.length; c++) {
                var ch = text[c].replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                if (marks[c] !== curMark) {
                    if (curMark > 0) html += '</span>';
                    curMark = marks[c];
                    if (curMark > 0) html += '<span class="' + cls[curMark] + '">';
                }
                html += ch;
            }
            if (curMark > 0) html += '</span>';
            return html;
        }

        function formatSpans(spans, label, cssClass) {
            if (!spans || spans.length === 0) return '<span class="' + cssClass + '"><span class="insp-meta-label">' + label + ':</span> <span style="opacity:0.5;font-style:italic">\u2014 none \u2014</span></span>';
            var parts = spans.map(function(s) {
                var val = s.v || 'detected';
                var typ = s.t || '';
                if (typ === 'pii') return val === 'detected' ? '<strong>PII detected</strong>' : val;
                return '<strong>' + typ + '</strong> \u2192 <code style="font-size:0.72rem;background:var(--bg);padding:1px 4px;border-radius:3px;border:1px solid var(--border)">' + val + '</code>';
            });
            return '<span class="' + cssClass + '"><span class="insp-meta-label">' + label + ':</span> ' + parts.join(' &middot; ') + '</span>';
        }

        function renderSample(uid, sample, pred) {
            var vc = pred.r.toLowerCase();
            var isDefend = curModel.indexOf('-defend') >= 0;
            var isLight = models[curModel] && models[curModel].light;
            var text, gt;
            if (!isDefend) {
                text = sample.x;
                gt = sample.g;
            } else if (isLight) {
                text = sample.dl || sample.d || sample.x;
                gt = sample.dlg || sample.dg || sample.g;
            } else {
                text = sample.d || sample.x;
                gt = sample.dg || sample.g;
            }
            var h = '<div class="insp-sample">';
            h += '<div class="insp-header">';
            h += '<span style="font-size:0.73rem;color:var(--text-muted)">Sample #' + uid + '</span>';
            h += '<span class="insp-badge insp-badge-' + vc + '">' + pred.r + '</span>';
            h += '</div>';
            h += '<div class="insp-text">' + highlightText(text, gt, pred.p) + '</div>';
            h += '<div class="insp-meta">';
            h += formatSpans(gt, 'Expected', 'insp-meta-expected');
            h += formatSpans(pred.p, 'Detected', 'insp-meta-detected');
            h += '</div></div>';
            return h;
        }

        var sectionDefs = [
            {segment: 'Negatives', label: 'Negatives', verdicts: ['FP', 'TN'], subLabels: {FP: 'False Positives \u2014 incorrectly flagged', TN: 'True Negatives \u2014 correctly ignored'}},
            {segment: 'Hard Negatives', label: 'Hard Negatives', verdicts: ['FP', 'TN'], subLabels: {FP: 'False Positives \u2014 incorrectly flagged', TN: 'True Negatives \u2014 correctly ignored'}},
            {segment: 'Positives', label: 'Positives (No Attack)', verdicts: ['FN', 'TP'], subLabels: {FN: 'False Negatives \u2014 PII missed', TP: 'True Positives \u2014 PII correctly detected'}},
            {segment: 'Direct Attack', label: 'Direct Attack Positives', verdicts: ['FN', 'TP'], subLabels: {FN: 'False Negatives \u2014 PII missed', TP: 'True Positives \u2014 PII correctly detected'}},
            {segment: 'Direct + Indirect', label: 'Direct + Indirect Attack Positives', verdicts: ['FN', 'TP'], subLabels: {FN: 'False Negatives \u2014 PII missed', TP: 'True Positives \u2014 PII correctly detected'}}
        ];

        var searchBox = document.getElementById('inspector-search');
        var searchQuery = '';
        if (searchBox) {
            var debounce = null;
            searchBox.addEventListener('input', function() {
                clearTimeout(debounce);
                debounce = setTimeout(function() {
                    searchQuery = searchBox.value.toLowerCase().trim();
                    renderInspector();
                }, 250);
            });
        }

        function matchesSearch(uid, sample, pred) {
            if (!searchQuery) return true;
            if (uid.toString().indexOf(searchQuery) >= 0) return true;
            if (sample.x.toLowerCase().indexOf(searchQuery) >= 0) return true;
            for (var i = 0; i < sample.g.length; i++) {
                if ((sample.g[i].v || '').toLowerCase().indexOf(searchQuery) >= 0) return true;
                if ((sample.g[i].t || '').toLowerCase().indexOf(searchQuery) >= 0) return true;
            }
            for (var i = 0; i < pred.p.length; i++) {
                if ((pred.p[i].v || '').toLowerCase().indexOf(searchQuery) >= 0) return true;
            }
            return false;
        }

        /* Track expanded limits per sub-section */
        var showLimits = {};

        function renderInspector() {
            var container = document.getElementById('inspector-sections');
            var preds = models[curModel] ? models[curModel].p : {};
            var uids = Object.keys(samples).sort(function(a,b){ return parseInt(a)-parseInt(b); });

            var grouped = {};
            for (var si = 0; si < sectionDefs.length; si++) {
                var sd = sectionDefs[si];
                grouped[sd.segment] = {};
                for (var vi = 0; vi < sd.verdicts.length; vi++) grouped[sd.segment][sd.verdicts[vi]] = [];
            }
            for (var u = 0; u < uids.length; u++) {
                var uid = uids[u];
                var s = samples[uid];
                var p = preds[uid];
                if (!p) continue;
                if (!matchesSearch(uid, s, p)) continue;
                if (grouped[s.c] && grouped[s.c][p.r]) {
                    grouped[s.c][p.r].push({uid: uid, sample: s, pred: p});
                }
            }

            var html = '';
            for (var si = 0; si < sectionDefs.length; si++) {
                var sd = sectionDefs[si];
                var grp = grouped[sd.segment];
                var total = 0;
                var countParts = [];
                for (var vi = 0; vi < sd.verdicts.length; vi++) {
                    var v = sd.verdicts[vi];
                    var n = grp[v].length;
                    total += n;
                    countParts.push(v + ': ' + n);
                }
                if (total === 0) continue;

                var isOpen = searchQuery ? ' open' : '';
                html += '<div class="insp-section' + isOpen + '">';
                html += '<div class="insp-section-header" onclick="this.parentElement.classList.toggle(\'open\')">';
                html += '<div style="display:flex;align-items:center;gap:0.5rem">';
                html += '<span class="insp-section-chevron">\u25B6</span>';
                html += '<span class="insp-section-title">' + sd.label + '</span>';
                html += '</div>';
                html += '<div class="insp-section-counts">';
                html += '<span>' + total + ' samples</span>';
                for (var ci = 0; ci < countParts.length; ci++) html += '<span>' + countParts[ci] + '</span>';
                html += '</div></div>';
                html += '<div class="insp-section-body">';

                for (var vi = 0; vi < sd.verdicts.length; vi++) {
                    var v = sd.verdicts[vi];
                    var items = grp[v];
                    if (items.length === 0) continue;
                    var limitKey = sd.segment + ':' + v + ':' + curModel;
                    var limit = showLimits[limitKey] || 20;
                    var pageSize = 20;
                    var page = showLimits[limitKey] || 0;
                    var totalPages = Math.ceil(items.length / pageSize);
                    var start = page * pageSize;
                    var end = Math.min(start + pageSize, items.length);
                    html += '<div class="insp-sub-header">' + sd.subLabels[v] + ' (' + items.length + ')</div>';
                    for (var i = start; i < end; i++) {
                        html += renderSample(items[i].uid, items[i].sample, items[i].pred);
                    }
                    if (totalPages > 1) {
                        html += '<div style="display:flex;justify-content:center;align-items:center;gap:0.5rem;margin:0.5rem 0">';
                        if (page > 0) {
                            html += '<button class="insp-page-btn" data-key="' + limitKey + '" data-page="' + (page - 1) + '" style="padding:0.25rem 0.8rem;border:1px solid var(--border);border-radius:4px;background:var(--bg);color:var(--text-muted);cursor:pointer;font-size:0.78rem">&laquo; Prev</button>';
                        }
                        html += '<span style="font-size:0.78rem;color:var(--text-muted)">Page ' + (page + 1) + ' / ' + totalPages + '</span>';
                        if (page < totalPages - 1) {
                            html += '<button class="insp-page-btn" data-key="' + limitKey + '" data-page="' + (page + 1) + '" style="padding:0.25rem 0.8rem;border:1px solid var(--border);border-radius:4px;background:var(--bg);color:var(--text-muted);cursor:pointer;font-size:0.78rem">Next &raquo;</button>';
                        }
                        html += '</div>';
                    }
                }
                html += '</div></div>';
            }

            container.innerHTML = html;

            /* Attach pagination handlers */
            var pageBtns = container.querySelectorAll('.insp-page-btn');
            for (var i = 0; i < pageBtns.length; i++) {
                pageBtns[i].addEventListener('click', function() {
                    var key = this.getAttribute('data-key');
                    var pg = parseInt(this.getAttribute('data-page'));
                    showLimits[key] = pg;
                    renderInspector();
                });
            }
        }

        renderInspector();
    }
});
"""

PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>'
