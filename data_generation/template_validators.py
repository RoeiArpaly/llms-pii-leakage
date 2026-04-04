"""Validates that LLM-generated text correctly contains (or omits) PII template
placeholders and does not leak real PII values through the templates.
"""
import re

from constants import PII_ENTITIES
from detectors.presidio import presidio_pii_analyzer


def contain_pii_template(text: str, contains_pii: bool):
    has_template = "{{" in text or "}}" in text
    extracted_params = (
        re.findall(pattern=r"{{(.*?)}}", string=text) if has_template else []
    )
    if contains_pii and not has_template:
        raise ValueError(f"Contains PII but no template found in the text:\n{text}")
    if not contains_pii and has_template:
        raise ValueError(f"No PII found, but a template was found in the text:\n{text}")
    if extracted_params and any(
        param not in PII_ENTITIES.values() for param in extracted_params
    ):
        raise ValueError(
            f"Invalid template found in the text.\nParameters: {extracted_params}"
        )
    if not extracted_params and has_template:
        raise ValueError("Invalid template found in the text.")
    results = presidio_pii_analyzer(text=text)
    if results:
        raise ValueError("Template contains PII")
