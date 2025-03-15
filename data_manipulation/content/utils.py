from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


def replacer(text: str, configs: list[dict]) -> str:
    """
    Replace the PII entity names with a replace_value.

    Parameters
    ----------
    text : str
        The text to replace the PII entities.
    configs : list[dict]
        A list of dictionaries containing the PII entity name, replace_value, and variations.

    Returns
    -------
    str

    """
    patterns_recognizers = {}
    operators = {}
    for config in configs:

        pii_entity = config["pii_entity"]
        variations = config["variations"]
        replace_value = config["replace_value"]

        regex = r"|".join(
            r"\b" + term.replace(" ", r"[\s_-]*") + r"\b" for term in variations
        )

        pattern = Pattern(name=pii_entity, regex=regex, score=1.0)
        patterns_recognizers[pii_entity] = pattern
        operators[pii_entity] = OperatorConfig(
            operator_name="replace", params={"new_value": replace_value}
        )

    # Analyzer
    analyzer = AnalyzerEngine()
    for recognizer in patterns_recognizers:
        patterns = patterns_recognizers[recognizer]
        pattern_recognizer = PatternRecognizer(supported_entity=recognizer, patterns=[patterns])
        analyzer.registry.add_recognizer(pattern_recognizer)
    results = analyzer.analyze(
        text=text, entities=list(patterns_recognizers.keys()), language="en"
    )

    # Anonymizer
    anonymizer = AnonymizerEngine()
    result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return result.text
