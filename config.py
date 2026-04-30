"""Central configuration for the PII detection evaluation framework.

All tuneable parameters (models, attacks, thresholds, match settings) are
defined as class attributes on Config and referenced throughout the pipeline.
"""


class Config:

    # Smoke-test mode. When True, all LLM calls return mock responses
    DRYRUN: bool = False

    # Dataset generation
    BULK_TARGET_N: int = 25_000
    BULK_WEIGHTS: tuple = (0.90, 0.05, 0.05)  # negative, positive, hard_negative
    BULK_CHECKPOINT_EVERY: int = 100
    BULK_WORKERS: int = 8
    BULK_MODEL: str = "gpt-4o-mini"
    RUN_NEURAL_ATTACKS: bool = False
    DETECTION_SAMPLE_N: dict = {
        "positives": None,                       # all clean positives
        "adv_positives_direct": 1_200,           # ~200 per fuzzy technique
        "adv_positives_direct_indirect": 6_000,  # ~100 per (fuzzy × adv) cell
        "negatives": 2_500,
        "hard_negatives": 750,
    }

    # Evaluation
    # Ordered lightest → heaviest: rule-based, transformers, SLMs, LLMs.
    MODELS: list = [
        # Rule-based (Presidio) — no model loading
        "presidio",
        "presidio-defend",
        "presidio-fuzzy",
        "presidio-fuzzy-defend",
        # Transformer NER (GLiNER, OpenAI Privacy Filter)
        "gliner",
        "gliner-defend",
        "gliner-nvidia",
        "gliner-nvidia-defend",
        "openai-privacy-filter",
        "openai-privacy-filter-defend",
        # Guard SLMs — by parameter count ascending
        "qwen-guard-0.6b",
        "qwen-guard-0.6b-defend",
        "llama-guard-3-1b",
        "llama-guard-3-1b-defend",
        "qwen-guard-4b",
        "qwen-guard-4b-defend",
        "nemotron-content-safety-4b",
        "nemotron-content-safety-4b-defend",
        "wildguard-7b",
        "wildguard-7b-defend",
        "llama-guard-3-8b",
        "llama-guard-3-8b-defend",
        "granite-guardian-8b",
        "granite-guardian-8b-defend",
        # Instruct SLM
        "llama-3.2-1b",
        "llama-3.2-1b-defend",
        # LLMs — API-based
        "gpt-4o-mini",
        "gpt-4o-mini-defend",
    ]
    ATTACKS: list = [
        "char_to_word",
        "chunking",
        "emojify",
        "homoglyph",
        "invisible_chars",
        "separators",
    ]
    CONTENT_ATTACKS: list = [
        "supportive_context",
        "prompt_injection_1",
        "prompt_injection_2",
        "prompt_injection_3",
        "prompt_injection_4",
        "prompt_injection_5",
        "prompt_injection_6",
        "affix_1",
        "affix_2",
        "affix_4",
    ]

    # PII evaluation level: "value", "type", or "both".
    MATCH_LEVEL: str = "both"
    # Fuzzy matching method: "exact", "subsequence", "difflib", "rapidfuzz", or "llm_judge".
    METHOD: str = "subsequence"
    LOGPROBS: bool = True  # if to compute perplexity scores

    # Guard
    PERPLEXITY_THRESHOLD: float = 1.5  # Threshold for perplexity to consider PII detected
    GLINER_THRESHOLD: float = 0.6  # GLiNER confidence threshold for PII Shield
