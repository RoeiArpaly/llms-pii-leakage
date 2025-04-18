import re


def smart_split(text: str) -> list[str]:
    """
    Splits the input string into a list of alphanumeric segments and
    individual non-alphanumeric characters.

    Parameters
    ----------
    text : str
        The input string to split.

    Returns
    -------
    list[str]
        A list of tokens split logically from the input string.
    """
    return re.findall(pattern=r"[a-zA-Z0-9]+|[^a-zA-Z0-9]", string=text)


def chunking(text: str) -> str:
    items = smart_split(text=text)
    if len(items) == 1:  # split in half
        item = items[0]
        if len(item) > 1:
            mid = len(item) // 2
            items = [f'"{item[:mid]}"', f'"{item[mid:]}"']
    else:
        items = [f'"{item}"' for item in items]
    return " + ".join(items)
