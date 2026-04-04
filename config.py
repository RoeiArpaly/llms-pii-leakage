"""Central configuration for the PII detection evaluation framework.

All tuneable parameters (models, attacks, thresholds, match settings) are
defined as class attributes on Config and referenced throughout the pipeline.
"""


class Config:

    # Mock LLM API calls (set to True to use mock responses instead of real API calls)
    MOCK_LLM: bool = True

    # Dataset generation
    SKIP_BASELINE: bool = True  # Skip baseline generation, reuse existing dataset
    PII_PROBABILITY: float = 0.5
    NUMBER_OF_SAMPLES: int = 20

    # Evaluation
    # Ordered lightest → heaviest: rule-based, transformers, SLMs, LLMs.
    MODELS: list = [
        # Rule-based (Presidio) — no model loading
        "presidio",
        "presidio-defend",
        "presidio-fuzzy",
        "presidio-fuzzy-defend",
        # Transformer NER (GLiNER) — small models, ONNX
        "gliner",
        "gliner-defend",
        "gliner-nvidia",
        "gliner-nvidia-defend",
        # SLMs — by parameter count ascending
        "qwen-guard-gen-0.6b",
        "qwen-guard-gen-0.6b-defend",
        "qwen-guard-stream-0.6b",
        "qwen-guard-stream-0.6b-defend",
        "llama-guard-3-1b",
        "llama-guard-3-1b-defend",
        # "nemotron-content-safety-4b",
        # "nemotron-content-safety-4b-defend",
        # "qwen-guard-gen-4b",
        # "qwen-guard-gen-4b-defend",
        # "qwen-guard-stream-4b",
        # "qwen-guard-stream-4b-defend",
        # "wildguard",
        # "wildguard-defend",
        # "llama-guard-3-8b",
        # "llama-guard-3-8b-defend",
        # LLMs — largest / API-based
        # "gpt-4o-mini",
        # "gpt-4o-mini-defend",

        # TODO: try SLM which is not guard, get perplexity and spans.
    ]
    ATTACKS: list = [
        "char_to_word",
        "chunking",
        "homoglyph",
        "emojify",
        "separators",
        "neural_prompt_to_prompt",
    ]
    CONTENT_ATTACKS: list = [
        "supportive_context",
        "prompt_injection_1",
        "prompt_injection_2",
        "affix_1",
        "affix_2",
    ]

    # PII evaluation level: "value", "type", or "both".
    MATCH_LEVEL: str = "both"
    # Fuzzy matching method: "exact", "subsequence", "difflib", "rapidfuzz", or "llm_judge".
    METHOD: str = "subsequence"
    LOGPROBS: bool = True  # if to compute perplexity scores

    # Guard
    PERPLEXITY_THRESHOLD: float = 1.000002  # Threshold for perplexity to consider PII detected
