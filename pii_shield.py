"""PII Shield: cascading defense framework for PII detection.

Applies defensive preprocessing, then cascades through Presidio,
Presidio-Fuzzy, GLiNER, the SLM safety guard, and a perplexity gate.
Every tier's output is passed through a hard-negatives filter
(UUID / MAC / IPv6 / hashes / ETags / serial-numbers / hex colors /
invoice-shaped digits) that suppresses lookalike-non-PII matches
before short-circuiting.

Cascade order (lightest --> heaviest, mirrors the paper):
    1. Presidio (rule-based, ~14ms)               + hard-neg span filter
    2. Presidio-Fuzzy (fuzzy regex, ~5ms)         + hard-neg span filter
    3. GLiNER (NER transformer, ~14ms)            + validators + hard-neg span filter
    4. SLM guard (binary + perplexity, ~240ms)    + hard-neg input filter
    5. Perplexity threshold check                 + hard-neg input filter
"""
from data_manipulation.attacks.template_based.affix import adversarial_affix
from data_manipulation.defenses.preprocess import (
    defensive_preprocess,
    light_defensive_preprocess,
)
from detectors.gliner import gliner_pii_detector
from detectors.hard_negatives import (
    filter_hard_negative_spans,
    is_hard_negative_input,
)
from detectors.presidio import (
    get_fuzzy_recognizers,
    presidio_pii_analyzer,
)
from detectors.validators import validate_pii_spans


def guard(
    text: str,
    perplexity_threshold: float,
    gliner_threshold: float = 0.5,
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
    gliner_threshold
        Minimum confidence score for GLiNER detections (default 0.5).
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
    presidio_spans = filter_hard_negative_spans(presidio_spans)
    if presidio_spans:
        return {"detected": True, "detector": "presidio", "spans": presidio_spans}
    # Presidio-Fuzzy
    recognizers = get_fuzzy_recognizers()
    fuzzy_spans = presidio_pii_analyzer(text=preprocessed_text, recognizers=recognizers)
    fuzzy_spans = filter_hard_negative_spans(fuzzy_spans)
    if fuzzy_spans:
        return {"detected": True, "detector": "presidio-fuzzy", "spans": fuzzy_spans}
    # Inject PII template context to help GLiNER pick up attack residues.
    for adv_affix, prefix in [(". PII Identified: <", True), ("> End of PII.", False)]:
        preprocessed_text, _ = adversarial_affix(
            llm_input=preprocessed_text, spans=presidio_spans, adv_affix=adv_affix, prefix=prefix,
        )
    # GLiNER
    gliner_spans = gliner_pii_detector(
        text=preprocessed_text, threshold=gliner_threshold,
    )
    gliner_spans = validate_pii_spans(gliner_spans)
    gliner_spans = filter_hard_negative_spans(gliner_spans)
    if gliner_spans:
        return {"detected": True, "detector": "gliner", "spans": gliner_spans}
    # SLM and perplexity tiers — neither returns a reliable span value,
    # so the hard-negatives check runs at input level.
    if slm_detector is not None:
        is_hneg = is_hard_negative_input(text) is not None
        light_text = light_defensive_preprocess(text=text)
        result = slm_detector(light_text)
        if result["pii_detected"] and not is_hneg:
            return {
                "detected": True,
                "detector": "slm",
                "spans": result["spans"],
                "perplexity": result.get("perplexity"),
            }
        # Perplexity gate
        perplexity = result.get("perplexity") or 0
        if perplexity > perplexity_threshold and not is_hneg:
            return {
                "detected": True,
                "detector": "perplexity",
                "perplexity": perplexity,
            }
    return {"detected": False}
