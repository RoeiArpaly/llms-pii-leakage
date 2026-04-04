"""Report display configuration: human-readable names, color palettes, marker
styles, and line styles for all models, attack types, and metrics used in
the evaluation report visualizations.
"""


DISPLAY_NAMES = {
    # Detectors
    "presidio": "Presidio",
    "presidio-defend": "Presidio + Shield",
    "presidio-fuzzy": "Presidio-Fuzzy",
    "presidio-fuzzy-defend": "Presidio-Fuzzy + Shield",
    "gliner": "GLiNER",
    "gliner-defend": "GLiNER + Shield",
    "gliner-nvidia": "GLiNER-NV",
    "gliner-nvidia-defend": "GLiNER-NV + Shield",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o-mini-defend": "GPT-4o-mini + Shield",
    "llama-guard-3-1b": "Llama Guard 1B",
    "llama-guard-3-1b-defend": "Llama Guard 1B + Shield",
    "llama-guard-3-8b": "Llama Guard 8B",
    "llama-guard-3-8b-defend": "Llama Guard 8B + Shield",
    "wildguard": "WildGuard",
    "wildguard-defend": "WildGuard + Shield",
    "nemotron-content-safety-4b": "Nemotron 4B",
    "nemotron-content-safety-4b-defend": "Nemotron 4B + Shield",
    "qwen-guard-gen-0.6b": "Qwen Guard 0.6B",
    "qwen-guard-gen-0.6b-defend": "Qwen Guard 0.6B + Shield",
    "qwen-guard-gen-4b": "Qwen Guard 4B",
    "qwen-guard-gen-4b-defend": "Qwen Guard 4B + Shield",
    "qwen-guard-stream-0.6b": "Qwen Stream 0.6B",
    "qwen-guard-stream-0.6b-defend": "Qwen Stream 0.6B + Shield",
    "qwen-guard-stream-4b": "Qwen Stream 4B",
    "qwen-guard-stream-4b-defend": "Qwen Stream 4B + Shield",
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
    "separators": "Separators",
    "reverse": "Reverse",
    "neural_prompt_to_prompt": "Neural P2P",
    # Content-level attacks
    "supportive_context": "Supportive Context",
    "supportive_context_affix_1": "Context + Affix I",
    "supportive_context_affix_2": "Context + Affix II",
    "supportive_context_prompt_injection_1": "Context + PI I",
    "supportive_context_prompt_injection_2": "Context + PI II",
    "supportive_context_affix_1_prompt_injection_2": "Context + Affix I + PI II",
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
    "presidio-defend":               "#74add1",
    "presidio-fuzzy":                "#313695",
    "presidio-fuzzy-defend":         "#abd9e9",
    "gliner":                        "#d73027",
    "gliner-defend":                 "#f46d43",
    "gliner-nvidia":                 "#a50026",
    "gliner-nvidia-defend":          "#fdae61",
    "gpt-4o-mini":                   "#1a9850",
    "gpt-4o-mini-defend":            "#66bd63",
    "llama-guard-3-1b":              "#8e44ad",
    "llama-guard-3-1b-defend":       "#bb8fce",
    "llama-guard-3-8b":              "#6c3483",
    "llama-guard-3-8b-defend":       "#a569bd",
    "wildguard":                     "#2e86c1",
    "wildguard-defend":              "#85c1e9",
    "nemotron-content-safety-4b":    "#b7950b",
    "nemotron-content-safety-4b-defend": "#d4ac0d",
    "qwen-guard-gen-0.6b":           "#e67e22",
    "qwen-guard-gen-0.6b-defend":    "#f5b041",
    "qwen-guard-gen-4b":             "#d35400",
    "qwen-guard-gen-4b-defend":      "#eb984e",
    "qwen-guard-stream-0.6b":        "#16a085",
    "qwen-guard-stream-0.6b-defend": "#48c9b0",
    "qwen-guard-stream-4b":          "#117864",
    "qwen-guard-stream-4b-defend":   "#76d7c4",
}

MODEL_MARKERS = {
    "presidio": "o", "presidio-defend": "s",
    "presidio-fuzzy": "D", "presidio-fuzzy-defend": "^",
    "gliner": "v", "gliner-defend": "<",
    "gliner-nvidia": "v", "gliner-nvidia-defend": "<",
    "gpt-4o-mini": ">", "gpt-4o-mini-defend": "p",
    "llama-guard-3-1b": "H", "llama-guard-3-1b-defend": "h",
    "llama-guard-3-8b": "H", "llama-guard-3-8b-defend": "h",
    "wildguard": "*", "wildguard-defend": "*",
    "nemotron-content-safety-4b": "+", "nemotron-content-safety-4b-defend": "+",
    "qwen-guard-gen-0.6b": "d", "qwen-guard-gen-0.6b-defend": "d",
    "qwen-guard-gen-4b": "8", "qwen-guard-gen-4b-defend": "8",
    "qwen-guard-stream-0.6b": "P", "qwen-guard-stream-0.6b-defend": "P",
    "qwen-guard-stream-4b": "X", "qwen-guard-stream-4b-defend": "X",
}

MODEL_LINESTYLES = {
    "presidio": "-", "presidio-defend": "--",
    "presidio-fuzzy": "-.", "presidio-fuzzy-defend": ":",
    "gliner": "-", "gliner-defend": "--",
    "gliner-nvidia": "-.", "gliner-nvidia-defend": ":",
    "gpt-4o-mini": "-", "gpt-4o-mini-defend": "--",
    "llama-guard-3-1b": "-", "llama-guard-3-1b-defend": "--",
    "llama-guard-3-8b": "-.", "llama-guard-3-8b-defend": ":",
    "wildguard": "-", "wildguard-defend": "--",
    "nemotron-content-safety-4b": "-.", "nemotron-content-safety-4b-defend": ":",
    "qwen-guard-gen-0.6b": "-", "qwen-guard-gen-0.6b-defend": "--",
    "qwen-guard-gen-4b": "-.", "qwen-guard-gen-4b-defend": ":",
    "qwen-guard-stream-0.6b": "-", "qwen-guard-stream-0.6b-defend": "--",
    "qwen-guard-stream-4b": "-.", "qwen-guard-stream-4b-defend": ":",
}

# Canonical ordering: Presidio → GLiNER → SLMs (small→large) → LLMs
_BASE_MODEL_ORDER = [
    "presidio",
    "presidio-fuzzy",
    "gliner",
    "gliner-nvidia",
    "llama-guard-3-1b",
    "llama-guard-3-8b",
    "wildguard",
    "nemotron-content-safety-4b",
    "qwen-guard-gen-0.6b",
    "qwen-guard-gen-4b",
    "qwen-guard-stream-0.6b",
    "qwen-guard-stream-4b",
    "gpt-4o-mini",
]

MODEL_ORDER = []
for _m in _BASE_MODEL_ORDER:
    MODEL_ORDER.append(_m)
    MODEL_ORDER.append(f"{_m}-defend")


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
