from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


_engine = None


def _build_engines(configs: list[dict]) -> tuple:
    """
    Build and cache the analyzer, anonymizer, and associated recognizers/operators.
    """
    patterns_recognizers = {}
    operators = {}
    for config in configs:
        pii_entity = config["pii_entity"]
        variations = config["variations"]
        replace_value = config["replace_value"]

        regex = r"|".join(r"\b" + term.replace(" ", r"[\s_-]*") + r"\b" for term in variations)
        pattern = Pattern(name=pii_entity, regex=regex, score=1.0)
        operator = OperatorConfig(operator_name="replace", params={"new_value": replace_value})
        patterns_recognizers[pii_entity] = pattern
        operators[pii_entity] = operator

    analyzer = AnalyzerEngine()
    analyzer.registry.recognizers = []
    for pii_entity, pattern in patterns_recognizers.items():
        pattern_recognizer = PatternRecognizer(supported_entity=pii_entity, patterns=[pattern])
        analyzer.registry.add_recognizer(pattern_recognizer)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer, operators


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
    global _engine
    cache_key = tuple((c["pii_entity"], c["replace_value"]) for c in configs)
    if not _engine or _engine[0] != cache_key:
        _engine = (cache_key, *_build_engines(configs=configs))
    _, _analyzer, _anonymizer, _operators = _engine

    entities = [config["pii_entity"] for config in configs]
    results = _analyzer.analyze(text=text, entities=entities, language="en")
    result = _anonymizer.anonymize(text=text, analyzer_results=results, operators=_operators)
    return result.text
