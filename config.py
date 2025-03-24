class Config:

    PII_PROBABILITY: float = 0.9
    NUMBER_OF_SAMPLES: int = 15
    MODELS: list = [
        "gpt-4o-mini-defend",
        "gpt-4o-mini",
        "gliner-defend",
        "gliner",
        "presidio-fuzzy-defend",
        "presidio-defend",
        "presidio-fuzzy",
        "presidio",
    ]
    MATCH_LEVEL: str = "both"
