from pandas import DataFrame


def dataframe_to_astar_html(
        data: DataFrame,
        title: str = "",
        caption: str = "",
        index: bool = False,
        highlight_cols: list[str] or str or bool = False,
        highlight_axis: int = 0,  # 1 or 0
        show_index_name: bool = True
) -> str:
    """
    Render a DataFrame as a centered, HTML table with title & caption.
    If the DataFrame has an index.name and show_index_name is True, it will be rendered as a
    vertical “y-axis” label to the left of the table.

    highlight_cols:
        - "auto" or None: highlights all numeric columns (default)
        - list of column names: highlights those columns
        - False: disables highlighting
    """
    if highlight_cols in (None, "auto"):
        highlight_cols = data.select_dtypes("number").columns.tolist()
    elif highlight_cols is False:
        highlight_cols = []

    css = [
        {
            "selector": "table.astar",
            "props": [
                ("font-family", "Times, 'Times New Roman', serif"),
                ("font-size", "11pt"),
                ("border-collapse", "collapse"),
                ("margin", "0"),
                ("width", "auto"),
                ("display", "table"),
            ],
        },
        {
            "selector": "thead tr:nth-child(2)",
            "props": [("display", "none")],
        },
        {
            "selector": "thead tr:nth-child(1) th",
            "props": [
                ("border-bottom", "2px solid #333"),
                ("text-align", "center"),
                ("padding", "0.25em 0.5em"),
            ],
        },
        {
            "selector": "tbody td",
            "props": [
                ("border-bottom", "1px solid #999"),
                ("text-align", "center"),
                ("padding", "0.2em 0.5em"),
            ],
        },
        {
            "selector": "caption",
            "props": [
                ("caption-side", "bottom"),
                ("font-size", "10pt"),
                ("padding-top", "0.3em"),
            ],
        },
    ]

    html = ['<div style="display: flex; flex-direction: column; align-items: center;">']
    if title:
        html.append(
            f'<div style="font-weight: bold; font-size: 12pt; margin-bottom: 0.4em;">'
            f'{title}</div>'
        )

    html.append(
        '<div style="display: flex; flex-direction: row; align-items: center;">'
    )

    if show_index_name and data.index.name:
        html.append(
            f"""
            <div style="writing-mode: vertical-rl;
                    text-orientation: mixed;
                    transform: rotate(180deg);
                    font-family: Times, 'Times New Roman', serif;
                    font-size: 11pt;
                    margin-right: 1em;
                    white-space: nowrap;">
            {data.index.name}
            </div>
            """
        )

    styler = (
        data.style
        .format("{:.2%}")
        .set_table_attributes('class="astar"')
        .set_table_styles(css)
    )
    if not index:
        styler = styler.hide(axis="index")
    if caption:
        styler = styler.set_caption(caption)

    if highlight_cols:
        styler = styler.highlight_max(
            subset=highlight_cols, axis=highlight_axis, props="font-weight: bold;"
        )

    html.append(styler.to_html())
    html.append('</div>')
    html.append('</div>')

    return "\n".join(html)
