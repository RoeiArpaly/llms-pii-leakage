import yaml

from importlib.resources import files

from constants import PII_ENTITIES
from utils import post_request_openai


PROMPTS_PATH = files("data_manipulation.attacks.neural_prompt_to_prompt").joinpath("prompts.yaml")

with PROMPTS_PATH.open("r") as f:
    PROMPTS = yaml.safe_load(f)


def llm_pii_fuzzer(
        llm_input: str,
        spans: list[dict],
        few_shots: bool = False,
        model: str = "gpt-4o-mini",
) -> tuple:
    if llm_input is None:
        return None, None

    required_pii = ", ".join([f"'{v}'" for v in PII_ENTITIES.values()])
    pii_instructions = f"\nThose are the only relevant PII types for this task: {required_pii}."

    content = PROMPTS["one_shot_rewrite"] + pii_instructions
    if few_shots:
        content += PROMPTS["few_shot_rewrite"]

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": content,
            },
            {
                "role": "user",
                "content": llm_input + str(spans),
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
                        "fuzzy_text": {
                            "type": "string",
                            "description": (
                                "The content input for the LLM model with PII manipulated."
                            ),
                        },
                        "pii_spans": {
                            "type": "array",
                            "description": (
                                "An array of objects representing "
                                "the detected PII spans in the text."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "value": {
                                        "type": "string",
                                        "description": (
                                            "The PII value identified in the text."
                                        ),
                                    },
                                    "start": {
                                        "type": "number",
                                        "description": (
                                            "The starting index of the PII span in the text."
                                        ),
                                    },
                                    "end": {
                                        "type": "number",
                                        "description": (
                                            "The ending index of the PII span in the text."
                                        ),
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": (
                                            f"The type of PII identified (e.g., {required_pii})."
                                        ),
                                    },
                                },
                                "required": ["value", "start", "end", "type"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["fuzzy_text", "pii_spans"],
                    "additionalProperties": False,
                },
            },
        },
        "temperature": 1,
        "max_tokens": 3_000,
    }
    json_schema = post_request_openai(data=data)
    return json_schema["fuzzy_text"], json_schema["pii_spans"]
