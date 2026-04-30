"""Chunking attack: splits PII values into quoted segments joined by ' + ' to
break contiguous pattern matching (e.g. "4111" + "1111" + "1111" + "1111").
"""
import re


def smart_split(text: str) -> list[str]:
    """
    Splits the input string into a list of alphanumeric segments and
    individual non-alphanumeric characters.
    """
    return re.findall(pattern=r"[a-zA-Z0-9]+|[^a-zA-Z0-9]", string=text)


def split_token(token: str) -> list[str]:
    """
    Splits numeric tokens into chunks of 3,
    borrowing one character if the last chunk would be length 1.

    Parameters
    ----------
    token : str
        The string to split.

    Returns
    -------
    list[str]
        A list of split string parts.
    """
    # Determine if we should split
    if token.isalpha():
        return [token]

    # Split into chunks of 3
    chunks = [token[i:i + 3] for i in range(0, len(token), 3)]
    # Borrow if last chunk is length 1
    if len(chunks) >= 2 and len(chunks[-1]) == 1:
        chunks[-2], chunks[-1] = chunks[-2][:-1], chunks[-2][-1] + chunks[-1]
    return chunks


def chunking(text: str) -> str:
    items = smart_split(text)  # Split into alnum / non-alnum tokens
    parts = []
    for item in items:
        if item.isalnum():
            parts.extend(split_token(item))
        else:
            parts.append(item)
    return " + ".join(f'"{part}"' for part in parts) if len(parts) > 1 else parts[0]
