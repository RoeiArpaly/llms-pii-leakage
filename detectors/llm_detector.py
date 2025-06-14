import yaml

from importlib.resources import files

from constants import PII_ENTITIES
from utils import post_request_openai


_prompts = None


def get_prompts():
    global _prompts
    if _prompts is None:
        prompts_path = files("detectors").joinpath("prompts.yaml")
        with prompts_path.open("r") as f:
            _prompts = yaml.safe_load(f)
    return _prompts


def llm_pii_detector(text: str, model: str = "gpt-4o-mini", logprobs: bool = False):
    """
    Detect PII in the LLM input.
    """
    if text is None:
        return

    content = get_prompts()["spans_detector"]
    pii_entities = list(PII_ENTITIES.values())

    json_schema = {
        "name": "llm_pii_detector_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["result", "spans"],
            "properties": {
                "result": {
                    "type": "object",
                    "required": ["pii_detected", "confidence"],
                    "properties": {
                        "pii_detected": {
                            "type": "boolean",
                            "description": (
                                f"Indicates whether at least one PII from the "
                                f"following PII entities: {pii_entities} "
                                f"was detected in the text."
                            )
                        },
                        "confidence": {
                            "type": "number",
                            "description": "A confidence parameter value between 0 and 1.",
                            "minimum": 0,
                            "maximum": 1
                        },
                    },
                    "additionalProperties": False,
                },
                "spans": {
                    "type": "array",
                    "description": "A list of dictionaries representing PII entities.",
                    "items": {
                        "type": "object",
                        "required": ["value", "start", "end", "type"],
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "The PII entity in a standard Presidio format."
                            },
                            "start": {
                                "type": "integer",
                                "description": "The start index of the PII entity."
                            },
                            "end": {
                                "type": "integer",
                                "description": "The end index of the PII entity."
                            },
                            "type": {
                                "type": "string",
                                "description": "The type of the PII entity.",
                                "enum": pii_entities,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    }

    data = {
        "model": model,
        "messages": [{"role": "system", "content": content}, {"role": "user", "content": text}],
        "response_format": {"type": "json_schema", "json_schema": json_schema},
        "temperature": 0,
        "max_tokens": 3_000,
    }
    if logprobs:
        data.update({"logprobs": True})
    response = post_request_openai(data=data, logprobs=logprobs)
    return response
