"""LLM-based prompt generation for PII detection datasets.

Uses OpenAI models to generate realistic user prompts with PII template
placeholders (e.g. {{credit_card_number}}) and hard-negative samples
containing PII-lookalike values.
"""
from random import choices

from numpy import random

from constants import (
    PII_ENTITIES,
    TOPICS,
)
from data_generation.template_validators import contain_pii_template
from utils import (
    load_prompts,
    post_request_openai,
    retry,
)


@retry(times=5)
def generate_hard_negative(model: str = "gpt-4o-mini") -> str:
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a text generator. Generate a realistic user message to an LLM "
                    "that does NOT contain real PII but DOES contain values that look similar "
                    "to PII. Include things like: invalid credit card numbers (failing Luhn), "
                    "SHA-256 hashes, UUIDs, tracking numbers, hex strings, test/placeholder "
                    "card numbers, API key formats, order IDs with digit patterns, MAC addresses, "
                    "or serial numbers. These should appear naturally in context. "
                    "Do NOT include real names, emails, phone numbers, SSNs, or valid IBANs."
                ),
            },
            {
                "role": "user",
                "content": f"Generate a message about: {', '.join(randomize_topics())}",
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "hard_negative_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "llm_input": {
                            "type": "string",
                            "description": "The generated text with PII-lookalike values.",
                        },
                    },
                    "required": ["llm_input"],
                    "additionalProperties": False,
                },
            },
        },
        "temperature": 1,
        "max_tokens": 10_000,
    }
    json_schema = post_request_openai(data=data)
    return json_schema["llm_input"]


def randomize_pii(r: float = 0.5, max_n: int = 10):
    """
    Randomly selects a number of PII entities based on a truncated geometric distribution.
    r - Decay rate, smaller = heavier bias toward small n
    max_n is the maximum number of PII entities to select.
    """
    optional_pii = list(PII_ENTITIES.values())

    # Sample from truncated geometric distribution
    weights = [(1 - r) * (r ** (n - 1)) for n in range(1, max_n + 1)]
    n = choices(population=range(1, max_n + 1), weights=weights, k=1)[0]
    return sorted(choices(population=optional_pii, k=n))


def randomize_topics():
    n = random.randint(low=1, high=3)
    return random.choice(a=TOPICS, size=n, replace=False).tolist()


@retry(times=5)
def generate_llm_input(contains_pii: bool, model: str = "gpt-4o-mini") -> str:
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": load_prompts("data_generation")["llm_input_generator"],
            },
            {
                "role": "user",
                "content": load_prompts("data_generation")["additional_instructions"].format(
                    contains_pii=contains_pii,
                    required_pii=randomize_pii() if contains_pii else None,
                    topics=randomize_topics(),
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "llm_input_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "llm_input": {
                            "type": "string",
                            "description": "The content input for the LLM model.",
                        },
                    },
                    "required": ["llm_input"],
                    "additionalProperties": False,
                },
            },
        },
        "temperature": 1,
        "max_tokens": 10_000,
    }
    json_schema = post_request_openai(data=data)
    llm_input = json_schema["llm_input"]
    contain_pii_template(text=llm_input, contains_pii=contains_pii)
    return llm_input
