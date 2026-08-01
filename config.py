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
        "presidio-fuzzy",
        # Transformer NER (GLiNER, OpenAI Privacy Filter)
        "gliner",
        "gliner-nvidia",
        "openai-privacy-filter",
        # Guard SLMs — by parameter count ascending
        "qwen-guard-0.6b",
        "llama-guard-3-1b",
        "qwen-guard-4b",
        "nemotron-content-safety-4b",
        "wildguard-7b",
        "llama-guard-3-8b",
        "granite-guardian-8b",
        # Instruct SLM
        "llama-3.2-1b",
        # LLMs — API-based
        "gpt-4o-mini",
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
        "pi_ceo_instruct",
        "pi_few_shot_safe",
        "pi_hypothetical",
        "pi_educational_framing",
        "pi_category_prime",
        "affix_redacted",
        "affix_ignore_pii",
        "affix_category_prime",
    ]

    # PII evaluation level: "value", "type", or "both".
    MATCH_LEVEL: str = "both"
    # Fuzzy matching method: "exact", "subsequence", "difflib", "rapidfuzz", or "llm_judge".
    METHOD: str = "subsequence"
    LOGPROBS: bool = True  # if to compute perplexity scores

    # Perplexity threshold for the standalone perplexity analysis.
    PERPLEXITY_THRESHOLD: float = 1.5
