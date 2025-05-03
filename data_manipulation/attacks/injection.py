from data_manipulation.attacks.template_based.affix import (
    AdversarialAffix,
    adversarial_affix,
)
from data_manipulation.attacks.template_based.prompt_injection import (
    PromptInjection,
    prompt_injection,
)
from data_manipulation.attacks.red_teaming import (
    chunking,
    emojify_pii,
    homoglyph,
    char_to_word,
    inject_separator,
)
from data_manipulation.attacks.red_teaming.content.supportive_context import supportive_context
from data_manipulation.attacks.red_teaming.pii.utils import fuzzy_pii_injection


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
        "char_to_word": char_to_word,
        "chunking": chunking,
        "emojify": emojify_pii,
        "homoglyph": homoglyph,
        "separators": inject_separator,
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
            result, new_spans = supportive_context(text=result, spans=new_spans)
        elif "affix" in technique:
            # Extract the affix number from the technique name
            affix_number = int(technique.split("_")[-1])
            adv_affix = AdversarialAffix.list_configs()[affix_number - 1]
            result, new_spans = adversarial_affix(
                llm_input=result, spans=new_spans, adv_affix=adv_affix.text, prefix=adv_affix.prefix
            )
        elif "prompt_injection" in technique:
            # Extract the affix number from the technique name
            prompt_injection_number = int(technique.split("_")[-1])
            adv_instructions = PromptInjection.list_configs()[prompt_injection_number - 1]
            result, new_spans = prompt_injection(
                llm_input=result, spans=new_spans, adv_instructions=adv_instructions.text
            )
        else:
            raise ValueError(f"Invalid technique: {technique}")
    return result, new_spans
