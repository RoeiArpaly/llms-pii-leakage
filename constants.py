DATASETS: list = ["baseline", "fuzzy", "fuzzy_adv"]
PII_ENTITIES: dict = {  # mapping of analyzer to generator entities
    "CREDIT_CARD": "credit_card_number",
    "IBAN_CODE": "iban",
    "US_SSN": "ssn",
    "PHONE_NUMBER": "phone_number",
}
FUZZY_TECHNIQUES: list[list] = [
    ["emojify"],
    ["homoglyph"],
    ["number_to_roman"],
    ["number_to_word"],
    ["reverse"],
    ["separators"],
]
ADV_CONTENT_TECHNIQUES: list[list] = [
    ["supportive_context"],
    ["affix"],
]
FUZZY_DATASET_COLS: list = [
    "input_id",
    "fuzzy_techniques",
    "llm_input",
    "pii_spans",
]
FUZZY_ADV_DATASET_COLS: list = [
    "input_id",
    "fuzzy_techniques",
    "adv_content_techniques",
    "llm_input",
    "pii_spans",
]
PREDICTION_DATASET_COLS: list = ["uid", "prediction", "spans_score"]
