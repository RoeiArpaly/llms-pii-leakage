from data_manipulation.defenses.preprocess import defensive_preprocess
from detectors.fuzzy_match import get_fuzzy_recognizers
from detectors.gliner_detector import gliner_pii_detector
from detectors.llm_detector import llm_pii_detector
from detectors.presidio_detector import presidio_pii_analyzer


def guard(text: str, perplexity_threshold: float) -> dict or None:
    # defensive preprocess
    preprocessed_text = defensive_preprocess(text=text)
    # Presidio
    presidio_spans = presidio_pii_analyzer(text=preprocessed_text)
    if presidio_spans:
        return {"detector": "presidio", "spans": presidio_spans}
    # GLiNER
    gliner_spans = gliner_pii_detector(text=preprocessed_text)
    if gliner_spans:
        return {"detector": "gliner", "spans": gliner_spans}
    # Fuzzy Presidio
    recognizers = get_fuzzy_recognizers()
    fuzzy_spans = presidio_pii_analyzer(text=preprocessed_text, recognizers=recognizers)
    if fuzzy_spans:
        return {"detector": "presidio-fuzzy", "spans": fuzzy_spans}
    # LLM
    predictions = llm_pii_detector(text=preprocessed_text, logprobs=True)
    llm_spans = predictions["structured_output"]["prediction"]
    if llm_spans:
        return {"detector": "llm", "spans": llm_spans}
    # Perplexity
    perplexity = predictions["perplexity"] or 1
    if perplexity > perplexity_threshold:
        return {"detector": "perplexity", "perplexity": perplexity}
