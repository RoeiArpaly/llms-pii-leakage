import yaml

from utils import post_request_openai


with open("detectors/prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)


def llm_pii_detector(text: str, mode: str = "text", model: str = "gpt-4o-mini"):
    """
    Detect PII in the LLM input.
    """
    if text is None:
        return

    if mode == "text":
        content = PROMPTS["text_detector"]
        prediction = {
            "type": "string",
            "description": "The text with the PII entities detected.",
        }

    elif mode == "spans":
        content = PROMPTS["spans_detector"]
        prediction = {
            "type": "array",
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
                    },
                },
                "required": ["value", "start", "end", "type"],
                "additionalProperties": False,
            },
        }

    else:
        raise ValueError("Invalid mode.")

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
