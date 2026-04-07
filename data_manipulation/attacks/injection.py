"""Orchestrates PII-level and content-level attack application.

pii_fuzzer applies character-level transformations (homoglyph, chunking, etc.)
to PII values. adversarial_content applies context-level attacks (supportive
context, prompt injection, affixes) around the PII.
"""
from data_manipulation.attacks.template_based.affix import (
    AdversarialAffix,
    adversarial_affix,
)
from data_manipulation.attacks.template_based.prompt_injection import (
    prompt_injection,
)
from data_manipulation.attacks.red_teaming import (
    char_to_word,
    chunking,
    emojify_pii,
    homoglyph,
    inject_separator,
    invisible_inject,
)
from data_manipulation.attacks.red_teaming.content.supportive_context import (
    supportive_context,
)
from data_manipulation.attacks.red_teaming.pii.utils import fuzzy_pii_injection

_PII_FUZZER_MAP = {
    "char_to_word": char_to_word,
    "chunking": chunking,
    "emojify": emojify_pii,
    "homoglyph": homoglyph,
    "invisible_chars": invisible_inject,
    "separators": inject_separator,
}


def pii_fuzzer(
    llm_input: str, spans: list[dict], chosen_techniques: list,
) -> tuple:
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

    result = llm_input
    new_spans = spans
    for technique in chosen_techniques:
        if technique in _PII_FUZZER_MAP:
            result, new_spans = fuzzy_pii_injection(
                text=result, spans=new_spans,
                fuzzy_func=_PII_FUZZER_MAP[technique],
            )
        elif technique == "baseline":
            continue
        else:
            raise ValueError(f"Invalid PII technique: {technique}")
    return result, new_spans


def adversarial_content(
    llm_input: str, spans: list[dict], chosen_techniques: list,
) -> tuple:
    if not chosen_techniques or not spans:
        return llm_input, spans

    result = llm_input
    new_spans = spans
    for technique in chosen_techniques:
        if technique == "supportive_context":
            result, new_spans = supportive_context(
                text=result, spans=new_spans,
            )
        elif "affix" in technique:
            n = int(technique.split("_")[-1])
            adv_affix = list(AdversarialAffix)[n - 1]
            result, new_spans = adversarial_affix(
                llm_input=result,
                spans=new_spans,
                adv_affix=adv_affix.text,
                prefix=adv_affix.prefix,
            )
        elif "prompt_injection" in technique:
            result, new_spans = prompt_injection(
                llm_input=result, spans=new_spans, technique=technique,
            )
        else:
            raise ValueError(f"Invalid content technique: {technique}")
    return result, new_spans
