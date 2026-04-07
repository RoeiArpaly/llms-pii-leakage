"""PII Shield: cascading defense framework for PII detection.

Applies defensive preprocessing, then cascades through Presidio, GLiNER,
fuzzy Presidio, SLM-based detection, and perplexity checking. Returns
on the first detector that finds PII.

Cascade order:
    1. Presidio (rule-based, ~0ms)
    2. GLiNER (NER transformer, ~10ms)
    3. Presidio-Fuzzy (rule-based with fuzzy recognizers, ~1ms)
    4. SLM guard (binary classification + perplexity, ~100ms)
    5. Perplexity threshold check (catches uncertain "safe" predictions)
"""
from data_manipulation.attacks.template_based.affix import adversarial_affix
from data_manipulation.defenses.preprocess import (
    defensive_preprocess,
    light_defensive_preprocess,
)
from detectors.gliner import gliner_pii_detector
from detectors.presidio import (
    get_fuzzy_recognizers,
    presidio_pii_analyzer,
)


def guard(
    text: str,
    perplexity_threshold: float,
    slm_detector=None,
) -> dict:
    """Cascading PII detection with multi-tier defense.

    Parameters
    ----------
    text
        The text to analyze for PII.
    perplexity_threshold
        If the SLM classifies text as safe but with perplexity above
        this threshold, the text is flagged as suspicious PII.
    slm_detector
        SLM detector function with signature (text) -> dict containing:
        - pii_detected (bool): whether PII was found
        - spans (list): detected PII spans
        - perplexity (float): model uncertainty (higher = less confident)
        If None, the SLM and perplexity steps are skipped.

    Returns
    -------
    dict
    """
    if not text:
        return {"detected": False}
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
    if slm_detector is not None:
        light_text = light_defensive_preprocess(text=text)
        result = slm_detector(light_text)
        if result["pii_detected"]:
            return {
                "detected": True,
                "detector": "slm",
                "spans": result["spans"],
                "perplexity": result.get("perplexity"),
            }
        # Perplexity
        perplexity = result.get("perplexity") or 0
        if perplexity > perplexity_threshold:
            return {
                "detected": True,
                "detector": "perplexity",
                "perplexity": perplexity,
            }
    return {"detected": False}
