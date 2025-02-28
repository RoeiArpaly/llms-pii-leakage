import yaml

from data_generation.data_validators import contain_pii_template
from utils import post_request_openai


with open("data_generation/prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)


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
    json_schema = post_request_openai(data=data)
    llm_input = json_schema["llm_input"]
    contain_pii_template(text=llm_input, contains_pii=contains_pii)
    return llm_input
