from presidio_analyzer import AnalyzerEngine

from constants import PII_ENTITIES


def presidio_pii_analyzer(text: str):

    if text is None:
        return []

    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language="en")

    relevant_results = [
        result for result in results if result.entity_type in PII_ENTITIES
    ]
    # Format the results in equivalent format to the Presidio Data Generator
    relevant_results = [
        {
            "value": text[result.start:result.end],
            "start": result.start,
            "end": result.end,
            "type": result.entity_type,
        }
        for result in relevant_results
    ]
    return relevant_results
