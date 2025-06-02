import json

from presidio_evaluator.data_generator import PresidioDataGenerator

from data_generation.pii_validators import (
    is_valid_credit_card,
    is_valid_iban,
)


_data_generator = None


def presidio_inject_pii(text: str):

    global _data_generator
    if not _data_generator:
        _data_generator = PresidioDataGenerator()

    fake_records = _data_generator.generate_fake_data(
        templates=[text],
        n_samples=1,
    )

    fake_records = list(fake_records)
    fake_records = json.loads(fake_records[0].toJSON())
    spans = json.loads(fake_records["spans"])
    spans = sorted(spans, key=lambda x: x["start"])  # Align with Presidio Analyzer
    for span in spans:
        if span["type"] == "credit_card_number":
            luhn_verify = is_valid_credit_card(card_number=span["value"])
            if not luhn_verify:
                raise ValueError(f"Invalid Luhn checksum. Credit Card: {span['value']}")
        if span["type"] == "iban":
            iban_verify = is_valid_iban(iban=span["value"])
            if not iban_verify:
                raise ValueError(f"Invalid IBAN: {span['value']}")
    return {"text": fake_records["fake"], "spans": spans}
