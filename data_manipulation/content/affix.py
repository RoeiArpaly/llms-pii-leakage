from copy import deepcopy
from enum import Enum
from typing import Union


class AdversarialContent(str, Enum):
    """
    Adversarial content options to be added to the PII.
    """
    ThisIsMyLuckyNumber = "NULL. and this is my lucky number:"
    Gibberish = "asdfasdfasdfasdf"
    Emojify = "👾👾👾👾👾"


def adversarial_affix(
    llm_input: str,
    spans: list[dict],
    adv_content: Union[AdversarialContent, str] = AdversarialContent.ThisIsMyLuckyNumber,
    prefix: bool = True,
    update_spans: bool = True,
) -> tuple:
    """
    Generate adversarial content by adding a prefix or a postfix to the PII.

    Span example:
        [{"value": "180087335977780", "start": 41, "end": 56, "type": "credit_card_number"}]
    """
    if llm_input is None:
        return llm_input, spans

    adv_content = adv_content + " " if prefix else " " + adv_content
    # Sort the spans in reverse order to avoid index conflicts
    spans = sorted(spans, key=lambda s: s.get("start"), reverse=True)
    new_spans = deepcopy(spans[::-1])  # Ascending order
    idx_offset = 0
    for span in spans:
        idx = span.get("start") if prefix else span.get("end")
        if idx is None:
            raise ValueError("Span is missing required 'start' or 'end' key.")
        llm_input = llm_input[:idx] + adv_content + llm_input[idx:]

    if update_spans:
        for new_span in new_spans:
            if prefix:
                idx_offset += len(adv_content)
                new_span["start"] += idx_offset
                new_span["end"] += idx_offset
            else:
                new_span["start"] += idx_offset
                new_span["end"] += idx_offset
                idx_offset += len(adv_content)
    return llm_input, new_spans
