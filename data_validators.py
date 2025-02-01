from presidio_analyzer import AnalyzerEngine


PRESIDIO_ANALYZER_GENERATOR_MAPPING = {
    "CREDIT_CARD": "credit_card_number",
    "IBAN_CODE": "iban",
    "US_SSN": "ssn",
    "PHONE_NUMBER": "phone_number",
}


def contain_pii_template(text: str) -> bool:

    pii_templates = PRESIDIO_ANALYZER_GENERATOR_MAPPING.values()
    if ("{{" in text) or ("}}" in text):
        if any(["{{" + template + "}}" in text for template in pii_templates]):
            return True
        else:
            raise ValueError("No valid PII template found in the text.")
    return False


def presidio_pii_analyzer(text: str):

    if text is None:
        return []

    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language="en")

    pii_templates = PRESIDIO_ANALYZER_GENERATOR_MAPPING.keys()
    relevant_results = [
        result for result in results if result.entity_type in pii_templates
    ]
    # Format the results in equivalent format to the Presidio Data Generator
    relevant_results = [
        {
            "value": text[result.start : result.end],
            "start": result.start,
            "end": result.end,
            "type": PRESIDIO_ANALYZER_GENERATOR_MAPPING[result.entity_type],
        }
        for result in relevant_results
    ]

    return relevant_results
