from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine

from constants import PII_ENTITIES


def presidio_pii_analyzer(text: str, nlp: bool = False):

    if text is None:
        return []

    nlp_engine = (
        SpacyNlpEngine(models=[{"lang_code": "en", "model_name": "en_core_web_lg"}])
        if nlp else None
    )

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en", "es"])
    results = analyzer.analyze(text=text, language="en")

    # Format the results in equivalent format to the Presidio Data Generator
    relevant_results = [
        {
            "value": text[result.start:result.end],
            "start": result.start,
            "end": result.end,
            "type": PII_ENTITIES[result.entity_type],
        }
        for result in results if result.entity_type in PII_ENTITIES
    ]
    return relevant_results
