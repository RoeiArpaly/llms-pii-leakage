"""Report display configuration: human-readable names, color palettes, marker
styles, and line styles for all models, attack types, and metrics used in
the evaluation report visualizations.
"""


DISPLAY_NAMES = {
    # Detectors
    "presidio": "Presidio",
    "presidio-fuzzy": "Presidio-Fuzzy",
    "gliner": "GLiNER",
    "gliner-nvidia": "GLiNER-NV",
    "gpt-4o-mini": "GPT-4o-mini",
    "llama-guard-3-1b": "Llama Guard 1B",
    "llama-guard-3-8b": "Llama Guard 8B",
    "wildguard-7b": "WildGuard",
    "nemotron-content-safety-4b": "Nemotron 4B",
    "qwen-guard-0.6b": "Qwen Guard 0.6B",
    "qwen-guard-4b": "Qwen Guard 4B",
    "granite-guardian-8b": "Granite Guardian 8B",
    "llama-3.2-1b": "Llama 3.2 1B",
    "openai-privacy-filter": "OpenAI Privacy Filter",
    # Categories
    "negative": "Negative",
    "hard_negative": "Hard Negative",
    "positive": "Positive",
    # PII-level attacks
    "baseline": "Baseline",
    "char_to_word": "Char-to-Word",
    "chunking": "Chunking",
    "emojify": "Emojify",
    "homoglyph": "Homoglyph",
    "invisible_chars": "Invisible Chars",
    "separators": "Separators",
    "reverse": "Reverse",
    "neural_prompt_to_prompt": "Neural P2P",
    # Content-level attacks
    # PI1 Authority · PI2 FewShot · PI3 Hypothetical · PI4 Educational
    # PI5 Category Prime · A1 Redacted · A2 Ignore · A3 Category Prime
    "supportive_context": "Supportive Context",
    "supportive_context_affix_redacted": "Context + Affix I",
    "supportive_context_affix_ignore_pii": "Context + Affix II",
    "supportive_context_affix_category_prime": "Context + Affix III",
    "supportive_context_pi_ceo_instruct": "Context + Authority",
    "supportive_context_pi_few_shot_safe": "Context + FewShot",
    "supportive_context_pi_hypothetical": "Context + Hypothetical",
    "supportive_context_pi_educational_framing": "Context + Educational",
    "supportive_context_pi_category_prime": "Context + Category Prime",
    "supportive_context_affix_redacted_pi_ceo_instruct": "Context + Affix I + Authority",
    "supportive_context_affix_ignore_pii_pi_few_shot_safe": "Context + Affix II + FewShot",
    "supportive_context_affix_category_prime_pi_category_prime": "Context + Affix III + Category Prime",
    "pi_educational_framing": "Educational Framing",
    # PII types
    "credit_card_number": "Credit Card",
    "iban": "IBAN",
    "ssn": "SSN",
    "phone_number": "Phone",
    "email": "Email",
    # Metrics
    "Precision": "Precision",
    "Recall": "Recall",
    "F1": "F1 Score",
}

MODEL_COLORS = {
    "presidio":                      "#4575b4",
    "presidio-fuzzy":                "#313695",
    "gliner":                        "#d73027",
    "gliner-nvidia":                 "#a50026",
    "openai-privacy-filter":         "#66bd63",
    "gpt-4o-mini":                   "#1a9850",
    "llama-guard-3-1b":              "#8e44ad",
    "llama-guard-3-8b":              "#6c3483",
    "wildguard-7b":                     "#2e86c1",
    "nemotron-content-safety-4b":    "#b7950b",
    "qwen-guard-0.6b":           "#e67e22",
    "qwen-guard-4b":             "#d35400",
    "granite-guardian-8b":       "#1b5e20",
    "llama-3.2-1b":              "#c62828",
}

MODEL_MARKERS = {
    "presidio": "o",
    "presidio-fuzzy": "D",
    "gliner": "v",
    "gliner-nvidia": "v",
    "openai-privacy-filter": "p",
    "gpt-4o-mini": ">",
    "llama-guard-3-1b": "H",
    "llama-guard-3-8b": "H",
    "wildguard-7b": "*",
    "nemotron-content-safety-4b": "+",
    "qwen-guard-0.6b": "d",
    "qwen-guard-4b": "8",
    "granite-guardian-8b": "P",
    "llama-3.2-1b": "X",
}

MODEL_LINESTYLES = {
    "presidio": "-",
    "presidio-fuzzy": "-.",
    "gliner": "-",
    "gliner-nvidia": "-.",
    "openai-privacy-filter": "--",
    "gpt-4o-mini": "-",
    "llama-guard-3-1b": "-",
    "llama-guard-3-8b": "-.",
    "wildguard-7b": "-",
    "nemotron-content-safety-4b": "-.",
    "qwen-guard-0.6b": "-",
    "qwen-guard-4b": "-.",
    "granite-guardian-8b": "-",
    "llama-3.2-1b": "-",
}

# Canonical ordering: Presidio → GLiNER → SLMs (small→large) → LLMs
_BASE_MODEL_ORDER = [
    "presidio",
    "presidio-fuzzy",
    "gliner",
    "gliner-nvidia",
    "openai-privacy-filter",
    "llama-guard-3-1b",
    "llama-guard-3-8b",
    "wildguard-7b",
    "nemotron-content-safety-4b",
    "qwen-guard-0.6b",
    "qwen-guard-4b",
    "granite-guardian-8b",
    "llama-3.2-1b",
    "gpt-4o-mini",
]

MODEL_ORDER = list(_BASE_MODEL_ORDER)


def model_sort_key(model: str) -> int:
    """Return sort index for a model name."""
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def sort_models(models) -> list[str]:
    """Sort model names in canonical order."""
    return sorted(models, key=model_sort_key)


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)
