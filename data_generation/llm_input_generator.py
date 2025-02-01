import json
import os

import requests

from data_generation.data_validators import contain_pii_template
from logger import logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_llm_input(contains_pii: bool):

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": """
                    You are an LLM input generator.
                    You are tasked to generate a random LLM input.
                    If you are requested to generate input with PII,
                    you must ensure that the generated content contains PII in the following format:
                    - Credit Card Number: {{credit_card_number}}
                    - IBAN: {{iban}}
                    - SSN: {{ssn}}
                    - Phone Number: {{phone_number}}

                    For example, "Give me a summary of the activities for {{credit_card_number}}
                    Dinning $1,200, Travel $800, Utilities $200, Other $500."

                    Do not generate PII which is not credit card number, IBAN, SSN, or phone number.
                    The results should be in English.
                """,
            },
            {
                "role": "user",
                "content": f"Generate content with{'' if contains_pii else 'out any'} PII",
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
            json_schema = json.loads(content)
            contain_pii_template(
                text=json_schema["llm_input"], contains_pii=contains_pii
            )
            json_schema["contains_pii"] = contains_pii
        except json.JSONDecodeError:
            logger.error(content)
            raise ValueError("Invalid JSON format.")
        return json_schema
    raise ValueError(f"Invalid response from OpenAI API.\n{response.text}")
