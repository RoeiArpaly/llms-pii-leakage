import random

from data_manipulation.content.emojify import emojify_pii_entity
from data_manipulation.content.affix import adversarial_affix
from data_manipulation.pii.emojify import emojify_pii
from data_manipulation.pii.number_to_word import number_to_word
from data_manipulation.pii.separators import inject_separator


def technique_sampler(techniques: list, n_techniques_upper=1) -> list:
    """

    Parameters
    ----------
    techniques : list
        The list of PII fuzzing techniques.
    n_techniques_upper : int
        The maximum number of techniques to apply to the text.

    Returns
    -------
    list

    """
    # randomly select 1 to n techniques
    chosen_techniques = random.sample(
        population=techniques,
        k=random.randint(1, n_techniques_upper),
    )
    return chosen_techniques


def pii_fuzzer(llm_input, spans, chosen_techniques):
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
        return

    result = llm_input
    for technique in chosen_techniques:
        if technique == "emojify":
            result = emojify_pii(text=result, spans=spans)
        elif technique == "number_to_word":
            result = number_to_word(text=result, spans=spans)
        elif technique == "separators":
            result = inject_separator(text=result, spans=spans)
        elif technique == "chunk_password":
            pass
        elif technique == "gibberish":
            pass
        elif technique == "random_case":
            pass
        else:
            raise ValueError(f"Invalid technique: {technique}")
    return result


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
