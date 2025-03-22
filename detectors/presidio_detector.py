from presidio_analyzer import AnalyzerEngine

from constants import PII_ENTITIES


_model = None


def get_presidio_model():
    global _model
    if _model is None:
        _model = AnalyzerEngine(supported_languages=["en"])
    return _model


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


def presidio_pii_analyzer(text: str, custom_recognizers: list = None):
    if text is None:
        return []

    analyzer = get_presidio_model()
    if custom_recognizers:
        for recognizer in custom_recognizers:
            analyzer.registry.add_recognizer(recognizer)

    results = analyzer.analyze(text=text, language="en")
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
    if custom_recognizers:
        relevant_results = filter_results(results=relevant_results)
    return relevant_results
