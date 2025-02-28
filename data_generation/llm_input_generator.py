import yaml

from importlib.resources import files

from numpy import random

from constants import (
    PII_ENTITIES,
    TOPICS,
)
from data_generation.data_validators import contain_pii_template
from utils import post_request_openai


PROMPTS_PATH = files("data_generation").joinpath("prompts.yaml")

with PROMPTS_PATH.open("r") as f:
    PROMPTS = yaml.safe_load(f)


def randomize_pii():
    optional_pii = list(PII_ENTITIES.values())
    n = random.randint(low=1, high=len(optional_pii) + 1)
    return random.choice(a=optional_pii, size=n, replace=False).tolist()


def randomize_topics():
    n = random.randint(low=1, high=3)
    return random.choice(a=TOPICS, size=n, replace=False).tolist()


def generate_llm_input(contains_pii: bool, model: str = "gpt-4o-mini") -> str:

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": PROMPTS["llm_input_generator"],
            },
            {
                "role": "user",
                "content": PROMPTS["additional_instructions"].format(
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
        "max_tokens": 3_000,
    }
    json_schema = post_request_openai(data=data)
    llm_input = json_schema["llm_input"]
    contain_pii_template(text=llm_input, contains_pii=contains_pii)
    return llm_input
