"""LLM-based prompt generation for PII detection datasets.

Uses OpenAI models to generate realistic user prompts with PII template
placeholders (e.g. {{credit_card_number}}) and hard-negative samples
containing PII-lookalike values.
"""
from random import choices

from numpy import random

from constants import (
    LENGTH_WEIGHTS,
    LENGTHS,
    LOOKALIKE_TYPES,
    LOWERCASE_OPENERS,
    PII_ENTITIES,
    SCENARIOS,
    TITLECASE_OPENERS,
    TONES,
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
    prompts = load_prompts("data_generation")
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": prompts["hard_negative_generator"],
            },
            {
                "role": "user",
                "content": prompts["hard_negative_instructions"].format(
                    opener=randomize_opener(),
                    scenario=randomize_scenario(),
                    topics=randomize_topics(),
                    tone=randomize_tone(),
                    length=randomize_length(),
                    required_lookalikes=randomize_lookalikes(),
                ),
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


_BUCKET_WEIGHTS = [0.70, 0.15, 0.13, 0.02]
_BUCKET_NAMES = ["single", "cluster", "multi", "many"]
_BUCKET_HINTS = {
    "single": (
        "The message is about a single piece of PII. The PII is the "
        "subject of the message — what the user is actually asking about."
    ),
    "cluster": (
        "The message naturally references several instances of the same "
        "PII type (e.g., an old/new pair, a typo correction, a "
        "sender/recipient pair, multiple records being compared). All "
        "instances must appear naturally in the same message."
    ),
    "multi": (
        "The message tells a coherent story that naturally requires the "
        "user to share multiple PII items together (e.g., updating "
        "account info, reporting fraud, dictating a contact card, filing "
        "a customer support ticket). The PII should feel necessary, not "
        "tacked on."
    ),
    "many": (
        "A richer scenario — perhaps a user filling out a complete "
        "account profile, providing identity-verification details, or "
        "describing a fraud incident with several touchpoints."
    ),
}


def randomize_pii() -> tuple[list[str], str]:
    """Pick PII placeholders for a positive sample with a realistic count
    distribution, plus a context hint describing the intended pattern.

    Returns
    -------
    types
        List of PII placeholder names (with possible repeats for the
        ``cluster`` bucket).
    context
        Short scenario hint passed to the LLM so it frames the prose
        coherently around the PII.
    """
    optional_pii = list(PII_ENTITIES.values())
    bucket = choices(population=_BUCKET_NAMES, weights=_BUCKET_WEIGHTS, k=1)[0]

    if bucket == "single":
        types = [random.choice(a=optional_pii).item()]
    elif bucket == "cluster":
        t = random.choice(a=optional_pii).item()
        n = random.randint(low=2, high=4)  # 2 or 3
        types = [t] * int(n)
    elif bucket == "multi":
        n = random.randint(low=2, high=4)  # 2 or 3 distinct types
        types = sorted(
            random.choice(a=optional_pii, size=n, replace=False).tolist(),
        )
    else:  # many
        n = min(int(random.randint(low=4, high=6)), len(optional_pii))
        types = sorted(
            random.choice(a=optional_pii, size=n, replace=False).tolist(),
        )

    return types, _BUCKET_HINTS[bucket]


def randomize_topics():
    n = random.randint(low=1, high=3)
    return random.choice(a=TOPICS, size=n, replace=False).tolist()


def randomize_lookalikes(min_n: int = 2, max_n: int = 4) -> list[str]:
    """Pick a random subset of PII-lookalike value types to require in the output."""
    n = random.randint(low=min_n, high=max_n + 1)
    return sorted(random.choice(a=LOOKALIKE_TYPES, size=n, replace=False).tolist())


def randomize_tone() -> str:
    return random.choice(a=TONES).item()


def randomize_length() -> str:
    """Pick a length label using a real-world-skewed distribution (95% short)."""
    return choices(population=LENGTHS, weights=LENGTH_WEIGHTS, k=1)[0]


def randomize_opener() -> str:
    """Pick a chat-style opener phrase. 50% from the lowercase pool, 50% from
    the titlecase pool — informal chat-message rate matches the user-facing
    "lowercase first letter" target."""
    pool = choices(
        population=[LOWERCASE_OPENERS, TITLECASE_OPENERS],
        weights=[0.5, 0.5],
        k=1,
    )[0]
    return random.choice(a=pool).item()


def randomize_scenario() -> str:
    """Pick a user-persona scenario to anchor the message in a realistic
    situation. Helps prevent template collapse at scale by adding a context
    axis the LLM can lean on."""
    return random.choice(a=SCENARIOS).item()


@retry(times=5)
def generate_llm_input(contains_pii: bool, model: str = "gpt-4o-mini") -> str:
    if contains_pii:
        required_pii, pii_context = randomize_pii()
    else:
        required_pii, pii_context = None, "N/A — message contains no PII."
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
                    required_pii=required_pii,
                    pii_context=pii_context,
                    opener=randomize_opener(),
                    scenario=randomize_scenario(),
                    topics=randomize_topics(),
                    tone=randomize_tone(),
                    length=randomize_length(),
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
