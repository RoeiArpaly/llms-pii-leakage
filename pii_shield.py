"""PII Shield: cascading defense framework for PII detection.

Applies defensive preprocessing, then cascades through Presidio, GLiNER,
fuzzy Presidio, LLM-based detection, and perplexity checking. Returns
on the first detector that finds PII.
"""
from data_manipulation.attacks.template_based.affix import adversarial_affix
from data_manipulation.defenses.preprocess import defensive_preprocess
from detectors.gliner import gliner_pii_detector
from detectors.llm import llm_pii_detector
from detectors.presidio import (
    get_fuzzy_recognizers,
    presidio_pii_analyzer,
)


def guard(text: str, perplexity_threshold: float) -> dict:
    """
    Description
    -----------
    This function analyzes the provided text for Personally Identifiable Information (PII).
    It employs a multi-layered approach to PII detection by applying the prevention module followed
    by the detection module.

    The detection module is cascading through various detection methods, including
    Presidio, GLiNER, fuzzy matching with Presidio, and a Large Language Model (LLM).
    It returns the detected PII spans or perplexity score
    if any PII is found. If no PII is detected, it returns None.

    Parameters
    ----------
    text: str
        The text to analyze for PII.
    perplexity_threshold: float
        The threshold for perplexity to consider PII detected.

    Returns
    -------
    dict
    """
    # Defensive preprocess
    preprocessed_text = defensive_preprocess(text=text)
    # Presidio
    presidio_spans = presidio_pii_analyzer(text=preprocessed_text)
    if presidio_spans:
        return {"detected": True, "detector": "presidio", "spans": presidio_spans}
    # Injecting PII template context
    for adv_affix, prefix in [(". PII Identified: <", True), ("> End of PII.", False)]:
        preprocessed_text, _ = adversarial_affix(
            llm_input=preprocessed_text, spans=presidio_spans, adv_affix=adv_affix, prefix=prefix,
        )
    # GLiNER
    gliner_spans = gliner_pii_detector(text=preprocessed_text)
    if gliner_spans:
        return {"detected": True, "detector": "gliner", "spans": gliner_spans}
    # Fuzzy Presidio
    recognizers = get_fuzzy_recognizers()
    fuzzy_spans = presidio_pii_analyzer(text=preprocessed_text, recognizers=recognizers)
    if fuzzy_spans:
        return {"detected": True, "detector": "presidio-fuzzy", "spans": fuzzy_spans}
    # LLM
    predictions = llm_pii_detector(text=preprocessed_text, logprobs=True)
    llm_spans = predictions["spans"]
    if llm_spans:
        return {"detected": True, "detector": "llm", "spans": llm_spans}
    # Perplexity
    perplexity = predictions["perplexity"] or 1
    if perplexity > perplexity_threshold:
        return {"detected": True, "detector": "perplexity", "perplexity": perplexity}
    return {"detected": False}
