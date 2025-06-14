DATASETS: list = ["baseline", "fuzzy", "fuzzy_adv"]
PII_ENTITIES: dict = {  # mapping of analyzer to generator entities
    "CREDIT_CARD": "credit_card_number",
    "IBAN_CODE": "iban",
    "US_SSN": "ssn",
    "PHONE_NUMBER": "phone_number",
    "EMAIL_ADDRESS": "email",
}
TOPICS: list = [
    "finance",
    "human resources",
    "daily usage",
    "engineering",
    "healthcare",
    "academia",
    "government",
    "entertainment",
    "sports",
    "technology",
    "business",
    "gaming",
    "advertisement",
    "social media",
    "marketing",
    "executive",
]
FUZZY_TECHNIQUES: list[list] = [
    ["char_to_word"],
    ["chunking"],
    ["emojify"],
    ["homoglyph"],
    ["separators"],
]
ADV_CONTENT_TECHNIQUES: list[list] = [
    ["supportive_context"],
    ["supportive_context", "affix_1"],
    ["supportive_context", "affix_2"],
    ["supportive_context", "affix_3"],
    ["supportive_context", "prompt_injection_1"],
    ["supportive_context", "prompt_injection_2"],
    ["supportive_context", "prompt_injection_3"],
]
BASELINE_DATASET_COLS: list = [
    "llm_input",
    "llm_input_defend",
    "pii_spans",
]
FUZZY_DATASET_COLS: list = [
    "input_id",
    "fuzzy_techniques",
    "llm_input",
    "llm_input_defend",
    "pii_spans",
]
FUZZY_ADV_DATASET_COLS: list = [
    "input_id",
    "fuzzy_techniques",
    "adv_content_techniques",
    "llm_input",
    "llm_input_defend",
    "pii_spans",
]
GLINER_INVALID_VALUES: list = [  # values that should not be detected by GLiNER
    "email",  # GLiNER struggles with emails
]
