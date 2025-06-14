from pandas import DataFrame


def dataframe_to_astar_html(
    data: DataFrame,
    title: str = "",
    caption: str = "",
    index: bool = False,
    highlight_cols: list[str] = None,
    show_index_name: bool = True
) -> str:
    """
    Render a DataFrame as a centered, HTML table with title & caption.
    If the DataFrame has an index.name and show_index_name is True, it will be rendered as a
    vertical “y-axis” label to the left of the table.
    """
    if highlight_cols is None:
        highlight_cols = data.select_dtypes("number").columns.tolist()

    # Booktabs-like CSS, hiding the in-table index-name row completely
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
        # remove the extra header row that only held the in-table index name
        {
            "selector": "thead tr:nth-child(2)",
            "props": [("display", "none")],
        },
        # styling for column headers (first row)
        {
            "selector": "thead tr:nth-child(1) th",
            "props": [
                ("border-bottom", "2px solid #333"),
                ("text-align", "center"),
                ("padding", "0.25em 0.5em"),
            ],
        },
        # body cells
        {
            "selector": "tbody td",
            "props": [
                ("border-bottom", "1px solid #999"),
                ("text-align", "center"),
                ("padding", "0.2em 0.5em"),
            ],
        },
        # caption styling
        {
            "selector": "caption",
            "props": [
                ("caption-side", "bottom"),
                ("font-size", "10pt"),
                ("padding-top", "0.3em"),
            ],
        },
    ]

    # Build HTML container
    html = ['<div style="display: flex; flex-direction: column; align-items: center;">']
    # Title above everything
    if title:
        html.append(
            f'<div style="font-weight: bold; font-size: 12pt; margin-bottom: 0.4em;">'
            f'{title}</div>'
        )

    # Start the row that holds the vertical index-name label + the table
    html.append(
        '<div style="display: flex; flex-direction: row; align-items: center;">'
    )

    # If index.name exists and should be shown, render it as rotated y-axis label
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

    # Build the Styler for the actual table
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
    styler = styler.highlight_max(subset=highlight_cols, props="font-weight: bold;")

    # Insert the table HTML
    html.append(styler.to_html())

    # Close row and container
    html.append('</div>')  # end flex row
    html.append('</div>')  # end flex column

    return "\n".join(html)
