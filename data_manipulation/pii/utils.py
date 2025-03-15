import random

from copy import deepcopy


def fuzzy_pii_injection(
        text: str,
        spans: list,
        fuzzy_func: callable,
        fuzzy_func_kwargs: dict = None,
        proba: float = 1.0,
        update_spans: bool = True,
) -> tuple:
    """Separate the text into components based on the spans."""
    components = []
    new_spans = deepcopy(spans)
    start = 0
    for i, span in enumerate(spans):
        end = span["start"]
        components.append(text[start:end])
        pii = text[span["start"]:span["end"]]
        if random.random() < proba:
            fuzzy_pii = fuzzy_func(pii, **(fuzzy_func_kwargs or {}))
            components.append(fuzzy_pii)
            if update_spans:
                new_spans[i]["value_fuzzy"] = fuzzy_pii
                new_spans[i]["end"] = new_spans[i]["start"] + len(fuzzy_pii)
        else:
            components.append(pii)
        start = span["end"]
    components.append(text[start:])
    return "".join(components), new_spans
