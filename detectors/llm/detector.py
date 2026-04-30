"""LLM-based PII detector using OpenAI's GPT models.

Sends text to an LLM with a structured JSON schema to detect PII spans.
Optionally extracts logprobs for the pii_detected token to compute
perplexity-based confidence scores.
"""
from constants import PII_ENTITIES
from utils import (
    load_prompts,
    post_request_openai,
)


def filter_pii_logprobs(logprobs_content: list[dict]) -> list[float]:
    """Extract logprobs of the true/false token following the 'pii_detected' key.

    Specific to OpenAI API structured JSON responses where the output
    contains a pii_detected boolean field.
    """
    key_sequence = ["pi", "i", "_detect", "ed", '":']
    n = len(key_sequence)

    for i in range(len(logprobs_content) - n):
        if [t["token"] for t in logprobs_content[i:i + n]] == key_sequence:
            value_token = logprobs_content[i + n]
            if value_token["token"] in ["true", "false"]:
                return [value_token["logprob"]]
            raise ValueError(
                f"Unexpected value token found: {value_token['token']}. "
                f"Expected 'true' or 'false'."
            )
    raise ValueError("No 'pii_detected' key found in logprobs content.")


def llm_pii_detector(text: str, model: str = "gpt-4o-mini", logprobs: bool = False) -> dict:
    """Detect PII spans in text using an LLM.

    Returns a dict with keys: result, spans, perplexity.
    """
    if text is None:
        return {
            "result": {"pii_detected": False, "predicted_proba": 0.0},
            "spans": [],
            "perplexity": None,
        }

    content = load_prompts("detectors.llm")["spans_detector"]
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
                    "required": ["pii_detected", "predicted_proba"],
                    "properties": {
                        "pii_detected": {
                            "type": "boolean",
                            "description": (
                                f"Indicates whether at least one PII from the "
                                f"following PII entities: {pii_entities} "
                                f"was detected in the text."
                            )
                        },
                        "predicted_proba": {
                            "type": "number",
                            "description": (
                                "A predicted_proba value between 0 and 1. "
                                "The higher amount/likelihood of PII entities detected from the "
                                f"following entities: {pii_entities}"
                                " - the higher the confidence score."
                            ),
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
