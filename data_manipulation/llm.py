import json
import os

import requests

from logger import logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def llm_pii_fuzzer(llm_input: str):
    if llm_input is None:
        return

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": """
You are a PII fuzzer.
You are tasked to manipulate the PII in the text.
You should rewrite the PII in a way that it cannot be identified by regex.
You must ensure that the PII is still identifiable by LLM or a human.
You can add some special letters or reorganize the structure.
However, you should be extra careful with emails or other
PII entities that can contain some special characters.

Use the following tactics:
1. use emojis (they will be ignored)
2. use special characters or letters as separators for numbers such as SSN and credit card
3. Replace numbers with letter representation such as 4 = four or 4 = D
4. For passwords that can contain almost all types of letters use emojis or
separate the password into chunks such as chunk 1: myp chunk2: assword123

Do not modify the text - only the PII.
Mix and match some of those techniques.
Do not obfuscate any of the info!
                """,
            },
            {
                "role": "user",
                "content": llm_input,
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pii_fuzzer_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "fuzzy_llm_input": {
                            "type": "string",
                            "description": (
                                "The content input for the LLM model with PII manipulated.",
                            ),
                        },
                    },
                    "required": ["fuzzy_llm_input"],
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
            json_schema = json.loads(content)["fuzzy_llm_input"]
        except json.JSONDecodeError:
            logger.error(content)
            raise ValueError("Invalid JSON format.")
        return json_schema
    raise ValueError(f"Invalid response from OpenAI API.\n{response.text}")
