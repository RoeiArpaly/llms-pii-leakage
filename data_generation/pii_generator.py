import json

from presidio_evaluator.data_generator import PresidioDataGenerator

from data_generation.data_validators import luhn_verify


def presidio_inject_pii(text: str):

    data_generator = PresidioDataGenerator()
    fake_records = data_generator.generate_fake_data(
        templates=[text],
        n_samples=1,
    )

    fake_records = list(fake_records)
    fake_records = json.loads(fake_records[0].toJSON())
    spans = json.loads(fake_records["spans"])
    spans = sorted(spans, key=lambda x: x["start"])  # Align with Presidio Analyzer
    for span in spans:
        if span["type"] == "credit_card_number":
            luhn_verify(string=span["value"])
    return {"text": fake_records["fake"], "spans": spans}
