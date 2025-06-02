import re


def smart_split(text: str) -> list[str]:
    """
    Splits the input string into a list of alphanumeric segments and
    individual non-alphanumeric characters.
    """
    return re.findall(pattern=r"[a-zA-Z0-9]+|[^a-zA-Z0-9]", string=text)


def split_token(token: str) -> list[str]:
    """
    Splits a single alphanumeric token into at least 2 chunks,
    using greedy logic with a preference for chunks of size 3.

    Parameters
    ----------
    token : str
        The string to split.

    Returns
    -------
    list[str]
        A list of split string parts.
    """
    length = len(token)
    if length == 1:
        return [token]
    # Force at least 2 chunks
    elif length < 6:
        mid = length // 2
        return [token[:mid], token[mid:]]
    chunks = []
    i = 0
    while length - i > 4:
        chunks.append(token[i:i + 3])
        i += 3
    chunks.append(token[i:])
    return chunks


def chunking(text: str) -> str:
    items = smart_split(text)
    if len(items) == 1 and items[0].isalnum():
        # One alphanumeric item — split if possible
        parts = split_token(items[0])
    else:
        # Multiple tokens — treat as atomic
        parts = items
    return " + ".join(f'"{part}"' for part in parts) if len(parts) > 1 else parts[0]
