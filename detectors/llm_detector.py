import yaml

from importlib.resources import files

from constants import PII_ENTITIES
from utils import post_request_openai


PROMPTS_PATH = files("detectors").joinpath("prompts.yaml")

with PROMPTS_PATH.open("r") as f:
    PROMPTS = yaml.safe_load(f)


def llm_pii_detector(text: str, model: str = "gpt-4o-mini"):
    """
    Detect PII in the LLM input.
    """
    if text is None:
        return

    content = PROMPTS["spans_detector"]
    prediction = {
        "type": "array",
        "description": "A list of dictionaries representing PII entities.",
        "items": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The PII entity in a standard Presidio format.",
                },
                "start": {
                    "type": "integer",
                    "description": "The start index of the PII entity.",
                },
                "end": {
                    "type": "integer",
                    "description": "The end index of the PII entity.",
                },
                "type": {
                    "type": "string",
                    "description": "The type of the PII entity.",
                    "enum": list(PII_ENTITIES.values()),
                },
            },
            "required": ["value", "start", "end", "type"],
            "additionalProperties": False,
        },
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "llm_pii_detector_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"prediction": prediction},
                    "required": ["prediction"],
                    "additionalProperties": False,
                },
            },
        },
        "temperature": 0,
        "max_tokens": 3_000,
    }
    json_schema = post_request_openai(data=data)
    return json_schema["prediction"]
