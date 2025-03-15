from data_manipulation.content import (
    adversarial_affix,
    supportive_context,
)
from data_manipulation.pii import (
    emojify_pii,
    homoglyph,
    number_to_roman,
    number_to_word,
    inject_separator,
    reverse_pii,
)
from data_manipulation.pii.utils import fuzzy_pii_injection


def pii_fuzzer(llm_input: str, spans: list[dict], chosen_techniques: list) -> tuple:
    """

    Parameters
    ----------
    llm_input : str
        The LLM input text.
    spans : list[dict]
        The PII spans in the text.
    chosen_techniques : list
        The PII fuzzing techniques to apply to the text.

    Returns
    -------
    str

    """

    if not chosen_techniques or not spans:
        return llm_input, spans

    mapping = {
        "emojify": emojify_pii,
        "homoglyph": homoglyph,
        "number_to_roman": number_to_roman,
        "number_to_word": number_to_word,
        "separators": inject_separator,
        "reverse": reverse_pii,
    }

    result = llm_input
    new_spans = spans
    for technique in chosen_techniques:
        if technique in mapping:
            result, new_spans = fuzzy_pii_injection(
                text=result,
                spans=new_spans,
                fuzzy_func=mapping[technique],
            )
        else:
            raise ValueError(f"Invalid technique: {technique}")

    return result, new_spans


def adversarial_content(llm_input: str, spans: list[dict], chosen_techniques: list) -> tuple:

    if not chosen_techniques or not spans:
        return llm_input, spans

    result = llm_input
    new_spans = spans
    for technique in chosen_techniques:
        if technique == "supportive_context":
            result, new_spans = supportive_context(text=result, spans=spans)
        elif technique == "affix":
            result, new_spans = adversarial_affix(llm_input=llm_input, spans=spans)
        elif technique == "rewrite":
            ...
        else:
            raise ValueError(f"Invalid technique: {technique}")
    return result, new_spans
