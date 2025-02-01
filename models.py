import json
import os
import random

import requests

from enum import Enum

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
            json_schema = json.loads(content)["llm_input"]
        except json.JSONDecodeError:
            logger.error(content)
            raise ValueError("Invalid JSON format.")
        return json_schema
    return response.text


def pii_fuzzer(llm_input: str):
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
                            "description": "The content input for the LLM model with PII manipulated.",
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
    return response.text


class AdversarialContent(str, Enum):
    """
    Detect the PII spans and add a postfix or a prefix of adversarial text.
    """

    Gibberish = "asdfasdfasdfasdf"
    Emojify = "👾👾👾👾👾"
    ThisIsMyLuckyNumber = "This is my lucky number: "


def adversarial_content_generator(
    llm_input: str,
    spans: list[dict],
    adv_content: AdversarialContent,
    prefix=True,
):
    """
    Detect the PII spans and add a postfix or a prefix of adversarial text.

    Span example:
        [{"value": "180087335977780", "start": 41, "end": 56, "type": "credit_card_number"}]

    """
    if llm_input is None:
        return

    spans = sorted(spans, key=lambda s: s.get("start"), reverse=True)
    for span in spans:
        # pii_type = span.get("type")
        idx = span.get("start") if prefix else span.get("end")
        if idx is None:
            raise ValueError("Span is missing required 'start' or 'end' key.")
        llm_input = llm_input[:idx] + adv_content + llm_input[idx:]
    return llm_input


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
    return response.text


def pii_fuzzer_type(n_techniques_upper=1):
    """

    Parameters
    ----------
    n_techniques_upper : int
        The maximum number of techniques to apply to the text.

    Returns
    -------
    list

    """
    techniques = [
        "emojify",
        # "number_to_word",
        # "special_characters",
    ]
    # randomly select 1 to n techniques
    chosen_techniques = random.sample(
        techniques, k=random.randint(1, n_techniques_upper)
    )
    return chosen_techniques


def pii_fuzzer_v2(llm_input, spans, chosen_techniques):
    """

    Parameters
    ----------
    llm_input : str
        The LLM input text.
    spans : list[dict]
        The PII spans in the text.
    chosen_techniques : list
        The PII fuzzing techniques to apply to the text.

    Returns
    -------
    str

    """

    from fuzzers.emojify import emojify_pii

    if not chosen_techniques:
        return

    result = llm_input
    for technique in chosen_techniques:
        if technique == "emojify":
            result = emojify_pii(text=llm_input, spans=spans)
        elif technique == "number_to_word":
            pass
        elif technique == "special_characters":
            pass
        elif technique == "chunk_password":
            pass
        elif technique == "gibberish":
            pass
        elif technique == "random_case":
            pass
        else:
            raise ValueError(f"Invalid technique: {technique}")
    return result


def adversarial_content(llm_input, chosen_techniques):
    from fuzzers.emojify import emojify_pii_entity

    if not chosen_techniques:
        return

    result = llm_input
    for technique in chosen_techniques:
        if technique == "emojify":
            result = emojify_pii_entity(text=llm_input)
        elif technique == "gibberish":
            ...
        else:
            raise ValueError(f"Invalid technique: {technique}")
    return result
