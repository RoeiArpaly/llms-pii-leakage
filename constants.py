PII_ENTITIES = {  # mapping of analyzer to generator entities
    "CREDIT_CARD": "credit_card_number",
    "IBAN_CODE": "iban",
    "US_SSN": "ssn",
    "PHONE_NUMBER": "phone_number",
}
FUZZY_TECHNIQUES = [
    "emojify",
    "number_to_word",
    "separators",
]
CONTENT_TECHNIQUES = [
    "emojify",
    "affix",
]
