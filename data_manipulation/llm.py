import yaml

from utils import post_request_openai


with open("data_manipulation/prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)


def llm_pii_fuzzer(llm_input: str, model: str = "gpt-4o-mini"):
    if llm_input is None:
        return

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": PROMPTS["pii_fuzzer"],
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
    json_schema = post_request_openai(data=data)
    return json_schema["fuzzy_llm_input"]
