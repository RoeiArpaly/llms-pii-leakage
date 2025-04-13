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
    if not spans:
        return text, []
    fuzzy_func_kwargs = fuzzy_func_kwargs or {}
    result = []
    new_spans = []
    input_idx = output_idx = 0
    for span in spans:
        start, end = span["start"], span["end"]
        result.append(text[input_idx:start])  # Add unchanged prefix
        output_idx += start - input_idx
        pii = text[start:end]
        if random.random() < proba:
            pii = fuzzy_func(pii, **fuzzy_func_kwargs)
        result.append(pii)
        new_span = deepcopy(span)
        if update_spans:
            new_span.update({
                "start": output_idx,
                "end": output_idx + len(pii),
                "value_fuzzy": pii,
            })
        new_spans.append(new_span)
        input_idx = end
        output_idx += len(pii)
    result.append(text[input_idx:])
    return "".join(result), new_spans
