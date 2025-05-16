class Config:

    # Dataset generation
    PII_PROBABILITY: float = 0.9
    NUMBER_OF_SAMPLES: int = 100

    # Evaluation
    MODELS: list = [
        # "gpt-4o-mini-defend",
        "gpt-4o-mini",
        # "gliner-defend",
        "gliner",
        # "presidio-fuzzy-defend",
        # "presidio-defend",
        # "presidio-fuzzy",
        "presidio",
    ]
    ATTACKS: list = [
        "char_to_word",
        "chunking",
        "homoglyph",
        "emojify",
        "separators",
        "reverse",
        "neural_prompt_to_prompt",
    ]
    CONTENT_ATTACKS: list = [
        "supportive_context",
        "prompt_injection_1",
        "prompt_injection_2",
        "prompt_injection_3",
        "affix_1",
        "affix_2",
        "affix_3",
        "neural_prompt_to_prompt",
    ]

    # PII evaluation level: "value", "type", or "both".
    MATCH_LEVEL: str = "both"
    # Fuzzy matching method: "exact", "subsequence", "difflib", "rapidfuzz", or "llm_judge".
    METHOD: str = "subsequence"
