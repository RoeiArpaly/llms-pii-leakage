from enum import Enum
from typing import Union


class AdversarialContent(str, Enum):
    """
    Detect the PII spans and add a postfix or a prefix of adversarial text.
    """

    Gibberish = "asdfasdfasdfasdf"
    Emojify = "👾👾👾👾👾"
    ThisIsMyLuckyNumber = "NULL. and this is my lucky number: "


def adversarial_affix(
    llm_input: str,
    spans: list[dict],
    adv_content: Union[AdversarialContent, str] = AdversarialContent.ThisIsMyLuckyNumber,
    prefix=True,
):
    """
    Detect the PII spans and add a postfix or a prefix of adversarial text.

    Span example:
        [{"value": "180087335977780", "start": 41, "end": 56, "type": "credit_card_number"}]

    """
    if llm_input is None:
        return

    adv_content = adv_content + " " if prefix else " " + adv_content
    spans = sorted(spans, key=lambda s: s.get("start"), reverse=True)
    for span in spans:
        # pii_type = span.get("type")
        idx = span.get("start") if prefix else span.get("end")
        if idx is None:
            raise ValueError("Span is missing required 'start' or 'end' key.")
        llm_input = llm_input[:idx] + adv_content + llm_input[idx:]
    return llm_input
