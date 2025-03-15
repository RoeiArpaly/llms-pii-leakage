class Config:

    PII_PROBABILITY: float = 0.9
    NUMBER_OF_SAMPLES: int = 15
    MODELS: list = ["gpt-4o-mini", "gliner", "presidio"]
    MATCH_LEVEL: str = "both"
