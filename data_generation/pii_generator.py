"""Synthetic PII injection using Presidio's sentence faker.

Takes a text template with PII placeholders and replaces them with realistic
fake PII values (credit cards, IBANs, SSNs, etc.), returning the filled text
and span metadata.
"""
import contextlib
import os
from functools import lru_cache

from presidio_evaluator.data_generator import PresidioSentenceFaker
from presidio_evaluator.data_generator.faker_extensions import PhoneNumberProviderNew

from constants import PII_ENTITIES
from data_generation.pii_validators import (
    is_valid_credit_card,
    is_valid_iban,
)

# Only the providers we actually need. PII templates use:
# {{credit_card_number}}, {{iban}}, {{ssn}}, {{phone_number}}, {{email}}.
# credit_card / iban / ssn / email are built into Faker for locale en_US, so
# the only custom provider we need is PhoneNumberProviderNew (multi-locale
# phone formats).
_ENTITY_PROVIDERS = [PhoneNumberProviderNew]


@lru_cache(maxsize=1)
def get_faker():
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
        return PresidioSentenceFaker(
            locale="en_US",
            lower_case_ratio=0.05,
            entity_providers=_ENTITY_PROVIDERS,
        )


def presidio_inject_pii(text: str):

    faker = get_faker()
    faker._sentence_templates = [text]
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
        samples = faker.generate_new_fake_sentences(num_samples=1)
    sample = samples[0]

    spans = sorted(
        [
            {
                "value": span.entity_value,
                "start": span.start_position,
                "end": span.end_position,
                "type": PII_ENTITIES.get(span.entity_type, span.entity_type),
            }
            for span in sample.spans if span.entity_type in PII_ENTITIES
        ],
        key=lambda x: x["start"],
    )
    for span in spans:
        if span["type"] == "credit_card_number":
            if not is_valid_credit_card(card_number=span["value"]):
                raise ValueError(f"Invalid Luhn checksum. Credit Card: {span['value']}")
        if span["type"] == "iban":
            if not is_valid_iban(iban=span["value"]):
                raise ValueError(f"Invalid IBAN: {span['value']}")
    return {"text": sample.full_text, "spans": spans}
