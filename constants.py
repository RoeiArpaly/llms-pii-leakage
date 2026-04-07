"""Project-wide constants: PII entity mappings, dataset categories, topic lists,
attack technique combinations, dataset column schemas, and GLiNER filter values.
"""
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
    ["invisible_chars"],
    ["separators"],
]
ADV_CONTENT_TECHNIQUES: list[list] = [
    ["supportive_context"],
    ["supportive_context", "affix_1"],
    ["supportive_context", "affix_2"],
    ["supportive_context", "prompt_injection_1"],
    ["supportive_context", "prompt_injection_2"],
    ["supportive_context", "affix_1", "prompt_injection_2"],
    ["supportive_context", "prompt_injection_3"],
    ["supportive_context", "prompt_injection_4"],
    ["supportive_context", "prompt_injection_5"],
]
DATASET_COLS: list = [
    "uid",
    "input_id",
    "category",
    "attack_target",
    "llm_input",
    "pii_spans",
]
GLINER_INVALID_VALUES: list = [  # values that should not be detected by GLiNER
    "email",  # GLiNER is triggered every time email is mentioned
    "e-addy",  # Slang for email
]
