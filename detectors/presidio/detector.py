"""Microsoft Presidio-based rule/regex PII detector.

Wraps AnalyzerEngine with project-specific entity filtering, overlap resolution,
IBAN validation, and optional custom recognizer support (e.g. fuzzy matching).
"""
from presidio_analyzer import AnalyzerEngine

from constants import PII_ENTITIES
from data_generation.pii_validators import is_valid_iban


_models = {
    "presidio": None,
    "presidio-defend": None,
}


def get_presidio_model(recognizers: list = None, use_cache: bool = True) -> AnalyzerEngine:
    model = "presidio"
    if recognizers:
        model = "presidio-defend"
    if _models[model] is not None and use_cache:
        return _models[model]
    engine = AnalyzerEngine(supported_languages=["en"])
    if recognizers:
        for recognizer in recognizers:
            engine.registry.add_recognizer(recognizer)
    if use_cache:
        _models[model] = engine
    return engine


def filter_results(results):
    """
    Filters PII detection results to retain only the highest-confidence match
    when there are overlapping start-end indexes.
    """
    if not results:
        return []
    # Sort results by confidence (higher first), then by start index
    results.sort(key=lambda result: (-result["score"], result["start"]))

    filtered = []
    taken_ranges = []
    for res in results:
        overlap = any(start <= res["end"] and res["start"] <= end for start, end in taken_ranges)
        if not overlap:
            filtered.append(res)
            taken_ranges.append((res["start"], res["end"]))  # Mark this range as taken
    return filtered


def presidio_pii_analyzer(
        text: str,
        recognizers: list = None,
        use_cache: bool = True,
        _analyzer_override: AnalyzerEngine = None,
) -> list:
    if text is None:
        return []

    has_custom_recognizers = bool(recognizers) or _analyzer_override is not None
    analyzer = _analyzer_override or get_presidio_model(recognizers=recognizers, use_cache=use_cache)
    results = analyzer.analyze(text=text, language="en", score_threshold=0.3)
    # Format the results in equivalent format to the Presidio Data Generator
    relevant_results = [
        {
            "value": text[result.start:result.end],
            "start": result.start,
            "end": result.end,
            "type": PII_ENTITIES[result.entity_type],
            "score": result.score,
            "recognizer": result.recognition_metadata["recognizer_name"],
        }
        for result in results if result.entity_type in PII_ENTITIES
    ]

    if has_custom_recognizers:
        relevant_results = filter_results(results=relevant_results)
        relevant_results = [
            res for res in relevant_results if
            not (res["type"] == "iban" and not is_valid_iban(res["value"]))
        ]
    return relevant_results
