import json
import os

import requests

from logger import logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def llm_pii_detector(text: str, mode="text"):
    """
    Detect PII in the LLM input.

    """
    if text is None:
        return

    if mode == "text":
        content = """
You are a PII detector.
Some of the PII in the text you receive may be altered,
making them unrecognizable by standard PII detection tools.
Your task is to rewrite these texts and restore the identifiable PII (back to the original text).
Think of using common decoding techniques or other methods to restore the PII.
You should not add any new information to the text.
"""
        prediction = {
            "type": "string",
            "description": "The text with the PII entities detected.",
        }

    elif mode == "spans":
        content = """
You are a PII detector.
You are tasked to detect all PII entities in the text.
You should identify the PII entities and their types, and provide the start and end indices.

For example:
* [{"value": "180087335977780", "start": 41, "end": 56, "type": "credit_card_number"}]
* [{"value": "DE89370400440532013000", "start": 65, "end": 87, "type": "iban_code"}]
* [{"value": "123456789", "start": 100, "end": 108, "type": "ssn"}]
* [{"value": "1234567890", "start": 120, "end": 130, "type": "phone_number"}]
"""
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
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": content,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "llm_pii_detector_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "prediction": prediction,
                    },
                    "required": ["prediction"],
                    "additionalProperties": False,
                },
            },
        },
        "temperature": 0,
        "max_tokens": 3_000,
    }

    response = requests.post(
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=data,
    )

    if response.status_code == 200:
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", {})
        )
        try:
            json_schema = json.loads(content)["prediction"]
        except json.JSONDecodeError:
            logger.error(content)
            raise ValueError("Invalid JSON format.")
        return json_schema
    raise ValueError(f"Invalid response from OpenAI API.\n{response.text}")
