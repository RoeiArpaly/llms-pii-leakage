import re

from constants import PII_ENTITIES


def contain_pii_template(text: str, contains_pii: bool):
    has_template = "{{" in text or "}}" in text
    extracted_params = (
        re.findall(pattern=r"{{(.*?)}}", string=text) if has_template else []
    )
    if contains_pii and not has_template:
        raise ValueError("Contains PII but no template found in the text.")
    if not contains_pii and has_template:
        raise ValueError("No PII found, but a template was found in the text.")
    if extracted_params and any(
        param not in PII_ENTITIES.values() for param in extracted_params
    ):
        raise ValueError(
            f"Invalid template found in the text.\nParameters: {extracted_params}"
        )
    if not extracted_params and has_template:
        raise ValueError("Invalid template found in the text.")


def luhn_verify(string):
    """
    Compute the Luhn checksum for the provided string of digits. Note this
    assumes the check digit is in place.
    """
    digits = list(map(int, string))
    odd_sum = sum(digits[-1::-2])
    even_sum = sum([sum(divmod(2 * d, 10)) for d in digits[-2::-2]])
    is_valid = (odd_sum + even_sum) % 10 == 0
    if not is_valid:
        raise ValueError(f"Invalid Luhn checksum. Credit Card: {string}")
