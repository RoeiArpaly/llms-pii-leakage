from data_manipulation.content import (
    adversarial_affix,
    emojify_pii_entity,
)
from data_manipulation.pii import (
    emojify_pii,
    number_to_roman,
    number_to_word,
    inject_separator,
)
from data_manipulation.pii.utils import fuzzy_pii_injection


def pii_fuzzer(llm_input, spans, chosen_techniques) -> tuple:
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

    if not chosen_techniques:
        return llm_input, spans

    mapping = {
        "emojify": emojify_pii,
        "number_to_roman": number_to_roman,
        "number_to_word": number_to_word,
        "separators": inject_separator,
        "chunk_password": None,
        "gibberish": None,
        "random_case": None,
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


def adversarial_content(llm_input, spans, chosen_techniques):

    if not chosen_techniques:
        return

    result = llm_input
    for technique in chosen_techniques:
        if technique == "emojify":
            result = emojify_pii_entity(text=result)
        elif technique == "affix":
            adversarial_affix(llm_input=llm_input, spans=spans)
        elif technique == "gibberish":
            ...
        else:
            raise ValueError(f"Invalid technique: {technique}")
    return result
